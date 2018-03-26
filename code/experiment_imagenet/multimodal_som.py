#! /usr/bin/env python
from collections import deque
import functools
import gc
import json
from pathlib import Path
import pickle
import random
import sys

import arrow
import click
from keras import optimizers
from keras import regularizers
from keras.layers import Dense, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Model
from keras.utils.np_utils import to_categorical
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin, BaseEstimator
import seaborn as sns
import somoclu
import spacy
import tensorflow as tf
from tensorflow import Summary as TfSummary
from tqdm import tqdm

from AlexNet import AlexNet, preprocess_image_batch


SumValue = TfSummary.Value
IMG_IN_SHAPE = (3, 227, 227)
IMG_OUT_SHAPE = 4096
WORD_OUT_SHAPE = 300
__file__ = Path(__file__).resolve()


class MultimodalSOM(BaseEstimator, TransformerMixin):
    def __init__(self, rows=10, cols=10, map_type='planar',
                 grid_type='rectangular', epochs=10):
        self.set_params(rows=rows, cols=cols, map_type=map_type,
                        grid_type=grid_type, epochs=epochs)
        self.nlp = spacy.load('en')
        self.som_model = create_som_model(self)
        self.alexnet, self.cnn_model = create_cnn_model()
        self.top_model = create_img_classifier()

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.rows = params['rows']
        self.cols = params['cols']
        self.map_type = params['map_type']
        self.grid_type = params['grid_type']
        self.epochs = params['epochs']
        self.params = params

    def train(self, X):
        self.som_model.train(X, epochs=self.epochs)

    def fit(self, X_train, y_train, X_val, y_val, batch_size=64,
            writer=None, batch_nr=0):
        indexes = np.arange(len(X_train))
        np.random.shuffle(indexes)
        X_train = X_train[indexes]
        y_train = y_train[indexes]
        y_val = to_categorical(y_val, num_classes=1000)
        train_results = []
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            y_batch = to_categorical(y_batch, num_classes=1000)
            batch_results = self.top_model.train_on_batch(X_batch, y_batch)
            train_results.append(batch_results)
            if writer is not None:
                summary = TfSummary(
                    value=[
                        SumValue(tag="batch loss",
                                 simple_value=batch_results[0]),
                        SumValue(tag="batch acc",
                                 simple_value=batch_results[1])
                    ]
                )
                writer.add_summary(summary, batch_nr)
        metrics_nr = len(self.top_model.metrics_names)
        train_results = list(zip(*train_results))
        train_results = [np.mean(train_results[i]) for i in range(metrics_nr)]
        val_results = self.top_model.evaluate(X_val, y_val, verbose=0)
        return train_results, val_results

    def transform(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def plot(self, fname):
        self.som_model.view_umatrix(bestmatches=True, filename=fname)

    def dump(self, fname):
        with open(f'{fname}_som.pickle', 'wb') as fp:
            pickle.dump(self.som_model, fp)
        with open(f'{fname}_params.json', 'w') as fp:
            json.dump(self.params, fp)
        self.top_model.save(f'{fname}_top.h5')

    def load(self, fname):
        with open(f'{fname}_params.json', 'r') as fp:
            self.set_params(**json.load(fp))
        with open(f'{fname}_som.pickle', 'rb') as fp:
            self.som_model = pickle.load(fp)
        self.nlp = spacy.load('en')
        self.top_model = load_model(f'{fname}_top.h5')

    def get_knn(self, dataset, target, k, metric, p):
        knn_dists = cdist([target], dataset, metric=metric, p=p)[0]
        knn_ids = np.argpartition(knn_dists, kth=k - 1)[:k]
        knn_ids = sorted(knn_ids, key=lambda idx: knn_dists[idx])
        return knn_ids

    def get_bmu(self, target=None, img_target=None, word_target=None,
                k=1, metric='som', p=None, ax=None):
        som_model = self.som_model
        codebook = som_model.codebook
        N, M, D = codebook.shape
        if metric == 'som':
            act_map = som_model.get_surface_state([target])
            bmu_ids = som_model.get_bmus(act_map)
        else:
            codebook = codebook.reshape(N * M, D)
            if target is not None:
                act_map = cdist([target], codebook, metric=metric, p=p)[0]
            elif img_target is not None:
                act_map = cdist([img_target], codebook[:, :IMG_OUT_SHAPE],
                                metric=metric, p=p)[0]
            elif word_target is not None:
                act_map = cdist([word_target], codebook[:, IMG_OUT_SHAPE:],
                                metric=metric, p=p)[0]
            bmu_ids = np.argpartition(act_map, kth=k - 1)[:k]
            bmu_ids = np.array(sorted(bmu_ids, key=lambda idx: act_map[idx]))
            bmu_ids = [np.unravel_index(bmu_id, (N, M)) for bmu_id in bmu_ids]
            act_map = 1 / act_map
        self.plot_activations(act_map, N, M, ax)
        return bmu_ids

    def plot_activations(self, activation_map, N, M, ax):
        if ax is not None:
            ax.imshow(activation_map.reshape(N, M), cmap='coolwarm')

    def get_knn_words(self, fname, k, vec2word, center=0, ax=None):
        som_model = self.som_model
        codebook = som_model.codebook
        N, M, D = codebook.shape

        get_image_features.cnn_model = self.cnn_model
        target = np.zeros(IMG_OUT_SHAPE + WORD_OUT_SHAPE)
        target[:IMG_OUT_SHAPE] = get_image_features(fname)

        metric = 'cosine'
        p = None

        bmu_ids = self.get_bmu(target=target, k=k, metric=metric, p=p, ax=ax)
        knn_ids = []
        for bmu_id in bmu_ids:
            word_features = codebook[bmu_id[0], bmu_id[1], IMG_OUT_SHAPE:]
            knn_id = self.get_knn(vec2word, word_features, 1,
                                  metric=metric, p=p)
            knn_ids.append(knn_id)

        return np.array(knn_ids).flatten()

    def get_knn_images(self, word, k, vec2img, center=0, ax=None):
        som_model = self.som_model
        codebook = som_model.codebook
        N, M, D = codebook.shape

        target = np.zeros(IMG_OUT_SHAPE + WORD_OUT_SHAPE)
        target[IMG_OUT_SHAPE:] = self.nlp.vocab[word].vector

        metric = 'cosine'
        p = None

        bmu_ids = self.get_bmu(target=target, k=1, metric=metric, p=p, ax=ax)
        knn_ids = []
        for bmu_id in bmu_ids:
            image_features = codebook[bmu_id[0], bmu_id[1], :IMG_OUT_SHAPE]
            knn_id = self.get_knn(vec2img, image_features, k,
                                  metric=metric, p=p)
            knn_ids.append(knn_id)

        return np.array(knn_ids).flatten()


def load_keras_image(fname):
    fname = Path(fname)
    x = preprocess_image_batch([fname], img_size=(256, 256),
                               crop_size=(227, 227), color_mode="rgb")
    return x


def project_images(x):
    if len(x.shape) < 4:
        x = np.expand_dims(x, axis=0)
    cnn_model = project_images.cnn_model
    return cnn_model.predict(x)


@functools.lru_cache()
def get_image_features(fname):
    project_images.cnn_model = get_image_features.cnn_model
    x = load_keras_image(fname)
    return project_images(x)


def create_img_classifier():
        input_len = IMG_OUT_SHAPE + WORD_OUT_SHAPE
        inputs = Input(shape=(input_len,),)
        x = BatchNormalization()(inputs)
        x = Dense(
            input_len, activation='relu',
            kernel_regularizer=regularizers.l1(0.01),
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        predictions = Dense(
            1000, activation='softmax',
            kernel_regularizer=regularizers.l2(0.01),
        )(x)

        # This creates a model that includes
        # the Input layer and three Dense layers
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=optimizers.SGD(lr=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['top_k_categorical_accuracy'])
        print(model.summary())
        return model


def create_som_model(parent):
    som_model = somoclu.Somoclu(parent.cols, parent.rows, kerneltype=0,
                                maptype=parent.map_type,
                                gridtype=parent.grid_type,
                                compactsupport=False,
                                initialization='random',
                                verbose=0)
    return som_model


def create_cnn_model():
    alexnet = AlexNet(weights='../alexnet_weights.h5')
    # dense_2 is the last fully connected layer, before softmax
    cnn_model = Model(inputs=alexnet.input,
                      outputs=alexnet.get_layer('dense_2').output)
    return alexnet, cnn_model


def generate_batches(features_prefix, train_data_dir, batch_size=1):
    train_data_dir = Path(train_data_dir)
    iterator = deque(set(train_data_dir.glob(f'**/{features_prefix}*.npy')))
    random.shuffle(iterator)
    feature_files_batches = []

    while iterator:
        feature_files_batch = []
        for _ in range(batch_size):
            if not iterator:
                break
            feature_files_batch.append(iterator.pop())
        if feature_files_batch:
            feature_files_batches.append(feature_files_batch)

    return feature_files_batches


def load_features(feature_files, val_indexes, test_size=0.2):
    X = []
    y = []
    for data_path in feature_files:
        data = np.load(data_path)
        X.extend(data[:, :-1])
        y.extend(data[:, -1].tolist())
    if X:
        X = np.array(X, dtype='float32')
        y = np.array(y)
        if val_indexes is None:
            sample_size = int(np.floor(len(X) * test_size))
            val_indexes = np.random.choice(np.arange(len(X)), size=sample_size,
                                           replace=False)
        X_val = X[val_indexes[val_indexes < len(X)]]
        y_val = y[val_indexes[val_indexes < len(X)]]
        train_indexes = np.arange(len(X))
        train_indexes = train_indexes[~np.isin(train_indexes, val_indexes)]
        X_train = X[train_indexes]
        y_train = y[train_indexes]
    if np.isnan(X).any():
        print(np.isnan(X).sum())
        raise Exception("NaN values in X")

    if np.isnan(y).any():
        print(np.isnan(y).sum())
        raise Exception("NaN values in y")
    return X_train, y_train, X_val, y_val, val_indexes


def train_som_model(mm_som, epochs, feature_prefix, batch_size,
                    train_data_dir):
    som_model = create_som_model(mm_som)
    tstart = arrow.now()
    tprev = arrow.now()
    mean_epoch_duration = 0
    for i in range(epochs):
        feature_files_batches = generate_batches(
            feature_prefix, train_data_dir=train_data_dir,
            batch_size=batch_size
        )
        iterator = tqdm(feature_files_batches, file=sys.stdout,
                        leave=False)
        for feature_files_batch in iterator:
            X, *_ = load_features(feature_files_batch, None, test_size=0)
            som_model.train(X, epochs=2)
            gc.collect()
            if np.isnan(som_model.codebook).any():
                raise Exception(f"Codebook contains NaN")

        tnow = arrow.now()
        elapsed = (tnow - tstart).total_seconds()
        epoch_duration = (tnow - tprev).total_seconds()
        mean_epoch_duration = ((mean_epoch_duration * i + epoch_duration) /
                               (i + 1))
        tprev = tnow
        print(f"Epoch: {i}. Elapsed: {elapsed:.4f}. "
              f"Mean duration: {mean_epoch_duration:.4f}")
    mm_som.som_model = som_model
    mm_som.dump('som_planar_rectangular')


def calculate_train_stats(train_results, val_results, mm_som):
    metrics_names = mm_som.top_model.metrics_names
    metrics_nr = len(metrics_names)

    train_results = list(zip(*train_results))
    train_results = [np.mean(train_results[i])
                     for i in range(metrics_nr)]
    val_results = list(zip(*val_results))
    val_results = [np.mean(val_results[i])
                   for i in range(metrics_nr)]
    epoch_results = {
        f'train_{metrics_names[i]}': train_results[i]
        for i in range(metrics_nr)
    }
    epoch_results.update({
        f'val_{metrics_names[i]}': val_results[i]
        for i in range(metrics_nr)
    })

    return epoch_results


def log_train_epoch(epoch_results, writer, i, mm_som, tstart, tprev,
                    mean_epoch_duration, epochs):
    metrics_names = mm_som.top_model.metrics_names
    trn_acc_key = 'train_top_k_categorical_accuracy'
    val_acc_key = 'val_top_k_categorical_accuracy'
    summary = TfSummary(
        value=[
            SumValue(tag="train loss", simple_value=epoch_results['val_loss']),
            SumValue(tag="train acc", simple_value=epoch_results[trn_acc_key]),
            SumValue(tag="val loss", simple_value=epoch_results['val_loss']),
            SumValue(tag="val acc", simple_value=epoch_results[val_acc_key]),
        ]
    )
    writer.add_summary(summary, i)

    tnow = arrow.now()
    elapsed = (tnow - tstart).total_seconds()
    epoch_duration = (tnow - tprev).total_seconds()
    mean_epoch_duration = ((mean_epoch_duration * i + epoch_duration) /
                           (i + 1))
    tprev = tnow
    print(f"{tnow.format('DD HH:mm:ss.SS')} - "
          f"Elapsed: {elapsed:0.4f}. "
          f"Mean epoch duration: {mean_epoch_duration:0.4f}")
    train_line = ' - '.join([
        f'{metrics_name}: {epoch_results[f"train_{metrics_name}"]:04.4f}'
        for metrics_name in metrics_names
    ])
    val_line = ' - '.join([
        f'{metrics_name}: {epoch_results[f"val_{metrics_name}"]:04.4f}'
        for metrics_name in metrics_names
    ])
    print(
        f"epoch: {i:04d}/{epochs} - train[{train_line}]\tval[{val_line}]"
    )
    return tprev, mean_epoch_duration


def train_top_model(mm_som, epochs, batch_size, early_stopping,
                    feature_prefix, train_data_dir, nn_batch_size):
    mm_som.top_model = create_img_classifier()
    val_indexes = None
    train_history = []
    non_decreasing_rounds = 0
    min_loss = np.infty
    print("Fitting top")
    writer = tf.summary.FileWriter(logdir='train_logs')
    tstart = arrow.now()
    tprev = arrow.now()
    mean_epoch_duration = 0
    batch_nr = 0
    for i in range(epochs):
        feature_files_batches = generate_batches(
            feature_prefix,  train_data_dir=train_data_dir,
            batch_size=batch_size
        )
        # iterator = tqdm(feature_files_batches, file=sys.stdout,
        #                 leave=False)
        train_results = []
        val_results = []
        for feature_files_batch in feature_files_batches:
            X_train, y_train, X_val, y_val, val_indexes =\
                load_features(feature_files_batch, val_indexes)
            batch_results =\
                mm_som.fit(X_train, y_train, X_val, y_val,
                           batch_size=nn_batch_size, writer=writer,
                           batch_nr=batch_nr)
            train_results.append(batch_results[0])
            val_results.append(batch_results[1])
            gc.collect()
            batch_nr += 1
        epoch_results = calculate_train_stats(train_results, val_results,
                                              mm_som)
        tprev, mean_epoch_duration = log_train_epoch(epoch_results, writer, i,
                                                     mm_som, tstart, tprev,
                                                     mean_epoch_duration,
                                                     epochs)
        train_history.append(epoch_results)

        current_loss = train_history[-1]['val_loss']
        if min_loss <= current_loss:
            non_decreasing_rounds += 1
        else:
            min_loss = current_loss
            non_decreasing_rounds = 0

        if non_decreasing_rounds >= early_stopping:
            print(
                f"Non decreasing val_loss for {early_stopping} "
                "rounds. Stopping"
            )
            break

    mm_som.dump('som_planar_rectangular')


@click.command()
@click.option('--batch_size', default=100)
@click.option('--train_data_dir', type=click.Path(exists=True, writable=True,
                                                  file_okay=False),
              default=f'/data/master_thesis/{__file__.parent.name}')
@click.option('--som_feature_prefix', type=str, default='features_som_')
@click.option('--top_feature_prefix', type=str, default='features_top_')
@click.option('--som_epochs', default=2)
@click.option('--top_epochs', default=10)
@click.option('--top_early_stopping', default=5)
@click.option('--top_batch_size', default=64)
@click.option('--load_model/--no_load_model', default=False)
@click.option('--train_som/--no_train_som', default=True)
@click.option('--train_top/--no_train_top', default=True)
def main(batch_size, train_data_dir, som_feature_prefix, top_feature_prefix,
         som_epochs, top_epochs, top_early_stopping, top_batch_size,
         load_model, train_som, train_top):
    train_data_dir = Path(train_data_dir)

    click.secho(f"batch_size: {batch_size}\n"
                f"train_data_dir: {train_data_dir}\n"
                f"som_epochs: {som_epochs}\n"
                f"top_epochs: {top_epochs}\n"
                f"top_early_stopping: {top_early_stopping}\n"
                f"top_batch_size: {top_batch_size}\n"
                f"load_model: {load_model}\n"
                f"train_som: {train_som}\n"
                f"train_top: {train_top}", color="blue")
    mm_som = MultimodalSOM(rows=32, cols=32, epochs=1)
    if load_model:
        mm_som.load('som_planar_rectangular')

    if train_som:
        train_som_model(mm_som=mm_som, epochs=som_epochs,
                        batch_size=batch_size, train_data_dir=train_data_dir,
                        feature_prefix=som_feature_prefix)

    if train_top:
        train_top_model(mm_som=mm_som, epochs=top_epochs,
                        batch_size=batch_size, train_data_dir=train_data_dir,
                        feature_prefix=top_feature_prefix,
                        early_stopping=top_early_stopping,
                        nn_batch_size=top_batch_size)


if __name__ == '__main__':
    sns.set(style='ticks', context='talk')
    main()
