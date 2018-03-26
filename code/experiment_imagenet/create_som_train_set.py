from collections import deque
import gc
import json
from pathlib import Path

import click
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from AlexNet import AlexNet, preprocess_image_batch
from statsrecorder import StatsRecorder

__file_path__ = Path(__file__).resolve()


def img_iterator(directory, wnid2idx, batch_size=10,
                 shape=(3, 227, 227), extension='.JPEG'):
    directory = Path(directory).resolve()
    iterator = deque(set(directory.glob(f'**/*{extension}')))
    gc.collect()

    while iterator:
        img_paths = []
        class_names = []
        class_ids = []
        img_names = []
        i = 0
        for i in range(batch_size):
            if not iterator:
                break
            img_path: Path = iterator.pop()
            img_paths.append(img_path)
            class_name = img_path.parent.name
            class_names.append(class_name)
            class_ids.append(wnid2idx[class_name])
            img_names.append(img_path.stem)
        if i != 0:
            yield (preprocess_image_batch(img_paths, img_size=(256, 256),
                                          crop_size=(227, 227),
                                          color_mode="rgb"),
                   class_names, class_ids, img_names)


def load_vec2word():
    nlp = spacy.load('en')
    vec2word_fname = Path('vec2word.csv')
    vec_cols = [f'v_{i:0>3}' for i in range(300)]
    if vec2word_fname.exists():
        vec2word = pd.read_csv(vec2word_fname)
    else:
        tokens = pd.read_table('/data/master_thesis/wnid2words.txt',
                               names=['wnid', 'word'], dtype=str)

        new_rows = []
        for i, w in tqdm(tokens.word.iteritems(), total=len(tokens)):
            words = str(w).split(',')
            new_rows.extend([(tokens.wnid[i], ww) for ww in words[1:]])
            tokens.word[i] = words[0]
        vec2word = tokens.append(pd.DataFrame(new_rows,
                                              columns=['wnid', 'word']))

        new_columns = []
        for _, w in tqdm(vec2word.word.iteritems(), total=len(vec2word)):
            new_columns.append(nlp(w).vector)

        agg_f = {c: 'mean' for c in vec_cols}
        agg_f['word'] = 'first'
        vec2word = vec2word\
            .assign(**{f'v_{i:0>3}': v
                       for i, v in enumerate(zip(*new_columns))})\
            .drop_duplicates(subset=['wnid'] + vec_cols)\
            .sort_values(['wnid'])\
            .groupby('wnid')\
            .agg(agg_f)\
            .reset_index()
        vec2word.to_csv(vec2word_fname, index=False)
        vec2word = vec2word[['wnid', 'word'] + sorted(vec2word.columns)[:-2]]
        vec2word.to_csv(vec2word_fname, index=False)
    if np.isnan(vec2word.loc[:, vec_cols].values).any():
        raise Exception("NaN values in vec2word")
    return vec2word, vec_cols


def load_wnid2idx(save_to_dir):
    wnid2idx_path = save_to_dir / 'wnid2idx.json'

    if wnid2idx_path.exists():
        with open(wnid2idx_path, 'r') as fp:
            wnid2idx = json.load(fp)
    else:
        wnid2idx = {
            w: i for i, w in
            enumerate(list(zip(*list(zip(
                *decode_predictions(np.eye(1000, 1000), top=1)
            ))[0]))[0])
        }
        with open(wnid2idx_path, 'w') as fp:
            json.dump(wnid2idx, fp, indent=4)

    return wnid2idx


def create_feature(cnn_model, alexnet, vec2word, vec_cols, save_to_dir,
                   wnid2idx, i, batch, class_names, class_ids):
    predictions = alexnet.predict(batch)
    predictions = decode_predictions(predictions, top=4)
    projections = cnn_model.predict(batch)
    features = []
    for j, (projection, prediction) in\
            enumerate(zip(projections, predictions)):
            true_class_name = class_names[j]
            class_id = class_ids[j]
            w2v = vec2word.loc[vec2word.wnid == true_class_name, vec_cols]
            w2v = w2v.values[0]
            features.append(np.r_[projection, w2v, class_id])

    features = np.array(features)
    if np.isnan(features).any():
        raise Exception(f"NaN values in features. i: {i}")

    np.save(save_to_dir / f'features_som_{i}.npy', features)


@click.command()
@click.option('--data_dir', type=click.Path(exists=True, writable=True,
                                            file_okay=False),
              default='/data/master_thesis/ILSVRC2012_img_train')
@click.option('--save_to_dir', type=click.Path(exists=True, writable=True,
                                               file_okay=False),
              default=f'/data/master_thesis/{__file_path__.parent.name}')
@click.option('--project/--no_project', default=True)
@click.option('--normalize/--no_normalize', default=True)
def main(data_dir, save_to_dir, project, normalize):
    click.secho(f"data_dir: {data_dir}\n"
                f"save_to_dir: {save_to_dir}\n"
                f"project: {project}\n"
                f"normalize: {normalize}",
                color="blue")
    vec2word, vec_cols = load_vec2word()
    alexnet = AlexNet(weights='alexnet_weights.h5')
    # dense_2 is the last fully connected layer, before softmax
    cnn_model = Model(inputs=alexnet.input,
                      outputs=alexnet.get_layer('dense_2').output)

    save_to_dir = Path(save_to_dir)

    wnid2idx = load_wnid2idx(save_to_dir)

    data_dir = Path(data_dir)
    extension = '.JPEG'
    batch_size = 15
    total_imgs = len(set(data_dir.glob(f'**/*{extension}'))) / batch_size
    total_imgs = int(np.ceil(total_imgs))
    if project:
        iterator = enumerate(
            tqdm(
                img_iterator(data_dir, wnid2idx,
                             shape=cnn_model.input_shape[1:],
                             extension=extension, batch_size=batch_size),
                total=total_imgs, desc="Projecting data"
            )
        )
        for i, (batch, class_names, class_ids, _) in iterator:
            create_feature(cnn_model, alexnet, vec2word, vec_cols, save_to_dir,
                           wnid2idx, i, batch, class_names, class_ids)

    if normalize:
        stats_rec = StatsRecorder()
        som_feature_files = list(save_to_dir.glob(f'**/features_som_*.npy'))
        for i, feature_path in enumerate(tqdm(som_feature_files,
                                              desc="Calculating μ and σ"),
                                         1):
            data = np.load(feature_path)
            features = data[:, :4396]
            stats_rec.update(features)
            if np.isnan(features.mean()).any():
                print(np.isnan(features.mean()).sum())
                raise Exception()

        click.secho(f"μ: {stats_rec.mean.mean()}\n"
                    f"σ: {stats_rec.std.mean()}",
                    color="blue")

        for i, feature_path in enumerate(tqdm(som_feature_files,
                                              desc="Normalizing")):
            data = np.load(feature_path)
            features = data[:, :4396]
            features -= stats_rec.mean
            stats_std = stats_rec.std
            stats_std[np.isnan(stats_std)] = 1
            features /= stats_std
            if np.isnan(features.mean()).any():
                print(np.isnan(features.mean()).sum())
                raise Exception()

            data[:, :4396] = features
            np.save(save_to_dir / f'features_som_norm_{i}.npy', data)


if __name__ == '__main__':
    main()
