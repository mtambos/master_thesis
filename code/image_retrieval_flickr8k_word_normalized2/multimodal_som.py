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
from keras.models import Model
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin, BaseEstimator
import seaborn as sns
import somoclu
import spacy
from tensorflow import Summary as TfSummary
from tqdm import tqdm

from AlexNet import AlexNet, preprocess_image_batch


SumValue = TfSummary.Value
IMG_IN_SHAPE = (3, 227, 227)
IMG_OUT_SHAPE = 4096
WORD_OUT_SHAPE = 300


class MultimodalSOM(BaseEstimator, TransformerMixin):
    def __init__(self, rows=10, cols=10, map_type='planar',
                 grid_type='rectangular', epochs=10):
        self.set_params(rows=rows, cols=cols, map_type=map_type,
                        grid_type=grid_type, epochs=epochs)
        self.nlp = spacy.load('en_core_web_md')
        self.som_model = create_som_model(self)
        self.alexnet, self.cnn_model = create_cnn_model()

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

    def load(self, fname):
        with open(f'{fname}_params.json', 'r') as fp:
            self.set_params(**json.load(fp))
        with open(f'{fname}_som.pickle', 'rb') as fp:
            self.som_model = pickle.load(fp)
        self.nlp = spacy.load('en_core_web_md')

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

    def get_knn_words(self, fname, k, vec2word, bmu_k=1, center=0, ax=None):
        som_model = self.som_model
        codebook = som_model.codebook
        N, M, D = codebook.shape

        get_image_features.cnn_model = self.cnn_model
        target = np.zeros(IMG_OUT_SHAPE + WORD_OUT_SHAPE)
        target[:IMG_OUT_SHAPE] = get_image_features(fname) / 4096

        metric = 'sqeuclidean'
        p = None

        bmu_ids = self.get_bmu(target=target, k=k, metric=metric, p=p, ax=ax)
        knn_ids = []
        for bmu_id in bmu_ids:
            word_features = codebook[bmu_id[0], bmu_id[1], IMG_OUT_SHAPE:]
            knn_id = self.get_knn(vec2word, word_features, 1,
                                  metric=metric, p=p)
            knn_ids.append(knn_id)

        return np.array(knn_ids).flatten()

    def get_knn_images(self, doc, k, vec2img, bmu_k=1, center=0, ax=None):
        som_model = self.som_model
        codebook = som_model.codebook
        N, M, D = codebook.shape

        target = np.zeros(IMG_OUT_SHAPE + WORD_OUT_SHAPE)
        doc = self.nlp(doc)
        target[IMG_OUT_SHAPE:] = doc.vector / 300

        metric = 'sqeuclidean'
        p = None

        bmu_ids = self.get_bmu(target=target, k=bmu_k, metric=metric, p=p, ax=ax)
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


def get_images_features(nouns_df):
    _, cnn_model = create_cnn_model()
    project_images.cnn_model = cnn_model

    image_features = (nouns_df.loc[:, ['fname']]
                              .drop_duplicates('fname')
                              .set_index('fname'))
    for i in range(IMG_OUT_SHAPE):
        image_features[f'vector{i}'] = 0

    step = 10
    total = len(image_features) // step
    for i in tqdm(range(0, len(image_features), step), total=total):
        fnames = image_features.index[i: i + step]
        X_img_train = np.zeros((len(fnames), *IMG_IN_SHAPE))
        for j, fname in enumerate(fnames):
            X_img_train[j] = load_keras_image(fname)
        projections = project_images(X_img_train)
        image_features.iloc[i: i + step, :] = projections

    return image_features


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
    for data_path in feature_files:
        data = np.load(data_path).item()
        data = data['batch']
        X.extend(data)
    if X:
        X = np.array(X, dtype='float32')
        if val_indexes is None:
            sample_size = int(np.floor(len(X) * test_size))
            val_indexes = np.random.choice(np.arange(len(X)), size=sample_size,
                                           replace=False)
    if np.isnan(X).any():
        print(np.isnan(X).sum())
        raise Exception("NaN values in X")

    return X


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
            X = load_features(feature_files_batch, None, test_size=0)
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


@click.command()
@click.option('--batch_size', default=100)
@click.option('--train_data_dir', type=click.Path(exists=True, writable=True,
                                                  file_okay=False),
              default=f'/data/master_thesis/{Path(__file__).parent.name}')
@click.option('--som_feature_prefix', type=str, default='features_som_')
@click.option('--som_epochs', default=2)
@click.option('--load_model/--no_load_model', default=False)
@click.option('--train_som/--no_train_som', default=True)
def main(batch_size, train_data_dir, som_feature_prefix,
         som_epochs, load_model, train_som):
    train_data_dir = Path(train_data_dir)

    click.secho(f"batch_size: {batch_size}\n"
                f"train_data_dir: {train_data_dir}\n"
                f"som_epochs: {som_epochs}\n"
                f"load_model: {load_model}\n"
                f"train_som: {train_som}\n")
    mm_som = MultimodalSOM(rows=100, cols=100, epochs=1)
    if load_model:
        mm_som.load('som_planar_rectangular')

    if train_som:
        train_som_model(mm_som=mm_som, epochs=som_epochs,
                        batch_size=batch_size, train_data_dir=train_data_dir,
                        feature_prefix=som_feature_prefix)


if __name__ == '__main__':
    sns.set(style='ticks', context='talk')
    main()
