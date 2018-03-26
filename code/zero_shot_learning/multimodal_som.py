#! /usr/bin/env python
# pylint: disable=too-many-arguments,R0914,R0915,C0111,C0103,R0902,R0903,R
import json
from pathlib import Path
import pickle
import warnings

import click
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import TransformerMixin, BaseEstimator
import seaborn as sns
import somoclu
import spacy
from tqdm import tqdm


IMG_IN_SHAPE = (3, 227, 227)
IMG_OUT_SHAPE = 4096
WORD_OUT_SHAPE = 300
__file__: Path = Path(__file__).resolve()  # type: ignore


def plot_activations(activation_map, N, M, ax):
    if ax is not None:
        activation_map = 1 / activation_map
        ax.imshow(activation_map.reshape(N, M), cmap='coolwarm')


def get_knns(dataset, targets, k, metric, p):
    if len(targets.shape) < 2:
        targets = np.expand_dims(targets, axis=0)
    knn_dists = cdist(targets, dataset, metric=metric, p=p)
    knn_ids = np.argsort(knn_dists, axis=1)[:, :k]
    return knn_ids


class MultimodalSOM(BaseEstimator, TransformerMixin):
    def __init__(self, rows=10, cols=10, map_type='planar',
                 grid_type='rectangular', epochs=10, load_cnn=True):
        self.set_params(rows=rows, cols=cols, map_type=map_type,
                        grid_type=grid_type, epochs=epochs)
        self.nlp = spacy.load('en_core_web_md')
        self.som_model = create_som_model(self)
        if load_cnn:
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

    def get_bmus(self, targets=None, img_targets=None, word_targets=None,
                 k=1, metric='som', p=None):
        som_model = self.som_model
        codebook = som_model.codebook
        N, M, D = codebook.shape
        codebook = codebook.reshape(N * M, D)

        if img_targets is not None:
            targets = img_targets
            codebook = codebook[:, :IMG_OUT_SHAPE]
        elif word_targets is not None:
            targets = word_targets
            codebook = codebook[:, IMG_OUT_SHAPE:]

        if len(targets.shape) < 2:
            targets = np.expand_dims(targets, axis=0)

        T, _ = targets.shape

        act_map = cdist(targets, codebook, metric=metric, p=p)
        bmu_ids = np.argpartition(act_map, kth=k - 1, axis=1)[:, :k]

        act_map_x_idx = np.broadcast_to(np.arange(T)[:, np.newaxis],
                                        bmu_ids.shape)
        bmu_act_map = act_map[act_map_x_idx, bmu_ids]

        bmu_sort_idx = np.argsort(bmu_act_map)
        bmu_ids_x_idx = np.broadcast_to(np.arange(T)[:, np.newaxis],
                                        bmu_sort_idx.shape)
        bmu_ids = bmu_ids[bmu_ids_x_idx, bmu_sort_idx]
        return bmu_ids, act_map

    def get_knn_words(self, k, vec2word, fnames=None, images=None,
                      images_features=None, kbmus=True):
        assert (
            (fnames is not None and
             images is None and
             images_features is None) or
            (fnames is None and
             images is not None and
             images_features is None) or
            (fnames is None and
             images is None and
             images_features is not None)
        )
        som_model = self.som_model
        codebook = som_model.codebook
        N, M, D = codebook.shape
        codebook = codebook.reshape(N * M, D)

        if fnames is not None:
            get_image_features.cnn_model = self.cnn_model
            images_features = get_image_features(*fnames)
        elif images is not None:
            project_images.cnn_model = self.cnn_model
            images_features = project_images(images)

        T, *_ = images_features.shape

        metric = 'cosine'
        p = None
        bmu_k = 1
        knn_k = k
        if kbmus:
            bmu_k = k
            knn_k = 1

        bmu_ids, _ = self.get_bmus(img_targets=images_features, k=bmu_k,
                                   metric=metric, p=p)
        word_features = codebook[bmu_ids, IMG_OUT_SHAPE:]
        word_features = word_features.reshape(T*bmu_k, WORD_OUT_SHAPE)
        knn_ids = get_knns(dataset=vec2word, targets=word_features,
                           k=knn_k, metric=metric, p=p).reshape(T, k)

        return knn_ids

    def get_knn_images(self, docs, k, vec2img, kbmus=True):
        som_model = self.som_model
        codebook = som_model.codebook
        N, M, D = codebook.shape
        codebook = codebook.reshape(N * M, D)
        T = len(docs)
        docs = [self.nlp(doc) for doc in docs]
        docs = np.array([doc.vector / doc.vector_norm for doc in docs])

        metric = 'cosine'
        p = None
        bmu_k = 1
        knn_k = k
        if kbmus:
            bmu_k = k
            knn_k = 1

        bmu_ids, _ = self.get_bmus(word_targets=docs, k=bmu_k, metric=metric,
                                   p=p)
        image_features = codebook[bmu_ids, :IMG_OUT_SHAPE]
        image_features = image_features.reshape(T*bmu_k, IMG_OUT_SHAPE)
        knn_ids = get_knns(dataset=vec2img, targets=image_features,
                           k=knn_k, metric=metric, p=p).reshape(T, k)

        return knn_ids


class ImgIterator(object):
    def __init__(self, imgs_dir, batch_size, epochs, cnn_model, nlp,
                 wnid2words=None):
        from keras.preprocessing.image import ImageDataGenerator
        self.imgs_dir = imgs_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.cnn_model = cnn_model
        self.nlp = nlp
        self.wnid2words = wnid2words
        self.gen = ImageDataGenerator()
        self.iterator = self.gen.flow_from_directory(
            self.imgs_dir, batch_size=self.batch_size, class_mode='sparse')
        self.reverse_class_indices = np.array([
            (v, k) for k, v in self.iterator.class_indices.items()
        ])
        project_images.cnn_model = cnn_model
        self.total_images = self.iterator.n
        self.batches_per_epoch = self.total_images//batch_size
        self.total_batches = self.batches_per_epoch * epochs
        self.current_epoch = 0
        self.current_batch_nr = 0

    def __next__(self):
        from AlexNet import preprocess_image_batch
        while True:
            if self.current_epoch >= self.epochs:
                raise StopIteration()
            if self.wnid2words is not None:
                ret_val = np.zeros((self.batch_size,
                                    IMG_OUT_SHAPE + WORD_OUT_SHAPE))
            else:
                ret_val = [np.zeros((self.batch_size, IMG_OUT_SHAPE)), '']

            try:
                batch, labels = next(self.iterator)
                batch = preprocess_image_batch(
                    image_arrays=batch, img_size=(256, 256),
                    crop_size=(227, 227), color_mode="rgb"
                )
                batch = project_images(batch)
                labels = self.reverse_class_indices[labels, 1]
                if self.wnid2words is not None:
                    labels = self.wnid2words.loc[labels, 'words']
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        labels = np.array([
                            self.nlp(w).vector/self.nlp(w).vector_norm
                            for w in labels
                        ])
                    labels[np.isnan(labels)] = 0
                    ret_val[:] = np.c_[batch, labels]
                else:
                    ret_val[0][:] = batch
                    ret_val[1] = labels
            except Exception as e:  # pylint: disable=W0703
                print(f'In ImgIterator.__next__: {e}')

            self.current_batch_nr += 1
            if self.current_batch_nr >= self.total_images // self.batch_size:
                self.current_batch_nr = 0
                self.current_epoch += 1

            return ret_val

    def __iter__(self):
        return self


def load_keras_image(fname):
    from AlexNet import preprocess_image_batch
    fname = Path(fname)
    x = preprocess_image_batch([fname], img_size=(256, 256),
                               crop_size=(227, 227), color_mode="rgb")
    return x


def project_images(x):
    if len(x.shape) < 4:
        x = np.expand_dims(x, axis=0)
    cnn_model = project_images.cnn_model
    return cnn_model.predict(x)


def get_image_features(*fnames):
    x = np.empty((len(fnames), *IMG_IN_SHAPE))
    for i, fname in enumerate(fnames):
        project_images.cnn_model = get_image_features.cnn_model
        x[i] = load_keras_image(fname)
    return project_images(x)


def create_som_model(parent):
    som_model = somoclu.Somoclu(parent.cols, parent.rows, kerneltype=0,
                                maptype=parent.map_type,
                                gridtype=parent.grid_type,
                                compactsupport=False,
                                initialization='random',
                                verbose=0)
    return som_model


def create_cnn_model():
    from keras.models import Model
    from AlexNet import AlexNet

    alexnet = AlexNet(weights='../alexnet_weights.h5')
    # dense_2 is the last fully connected layer, before softmax
    cnn_model = Model(inputs=alexnet.input,
                      outputs=alexnet.get_layer('dense_2').output)
    return alexnet, cnn_model


def train_som_model(mm_som, epochs, wnid2words, batch_size, imgs_dir,
                    save_state_per_epoch):
    som_model = create_som_model(mm_som)
    iterator = ImgIterator(imgs_dir, batch_size, epochs, mm_som.cnn_model,
                           mm_som.nlp, wnid2words)
    for i, batch in enumerate(tqdm(iterator, total=iterator.total_batches)):
        som_model.train(batch, epochs=2)
        if save_state_per_epoch and i % iterator.batches_per_epoch == 0:
            mm_som.som_model = som_model
            mm_som.dump('som_planar_rectangular')
            print(f'State saved, {i} batches processed.')
        if np.isnan(som_model.codebook).any():
            raise Exception(f"Codebook contains NaN")

    mm_som.som_model = som_model
    mm_som.dump('som_planar_rectangular')


@click.command()
@click.option('--batch_size', default=100)
@click.option('--imgs_dir', type=click.Path(exists=True, file_okay=False),
              default=f'/data/master_thesis/ILSVRC2012_img_train')
@click.option('--wnid_map_file', type=click.Path(exists=True, dir_okay=False),
              default=f'/data/master_thesis/wnid2words.txt')
@click.option('--epochs', default=2)
@click.option('--load_model/--no_load_model', default=False)
@click.option('--train/--no_train', default=True)
def main(batch_size, imgs_dir, wnid_map_file,
         epochs, load_model, train):
    imgs_dir = Path(imgs_dir)
    wnid_map_file = Path(wnid_map_file)

    click.secho(f"batch_size: {batch_size}\n"
                f"imgs_dir: {imgs_dir}\n"
                f"wnid_map_file: {wnid_map_file}\n"
                f"epochs: {epochs}\n"
                f"load_model: {load_model}\n"
                f"train: {train}\n")
    mm_som = MultimodalSOM(rows=100, cols=100, epochs=1)
    wnid2words = pd.read_table(wnid_map_file, names=['wnid', 'words'],
                               index_col='wnid')
    if load_model:
        mm_som.load('som_planar_rectangular')

    if train:
        train_som_model(mm_som=mm_som, epochs=epochs,
                        batch_size=batch_size, imgs_dir=imgs_dir,
                        wnid2words=wnid2words, save_state_per_epoch=True)


if __name__ == '__main__':
    sns.set(style='ticks', context='talk')
    main()  # pylint: disable=E1120
