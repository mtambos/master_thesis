from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from tqdm import tqdm

from multimodal_som import get_image_features, create_cnn_model


IMG_IN_SHAPE = (3, 227, 227)
IMG_OUT_SHAPE = 4096
WORD_OUT_SHAPE = 300
__file__ = Path(__file__).resolve()


def load_captions(tokens_file):
    captions = pd.read_table(tokens_file, names=['fname', 'doc'])
    captions['fname'] = captions.fname.str.replace(r'[#.]\d', '')
    return captions, np.unique(captions.fname)


def iter_nouns(captions, img_dir):
    img_dir = Path(img_dir)
    nlp = spacy.load('en')
    nouns_s = captions.doc.apply(nlp).apply(
        lambda d: np.array([t for t in d if t.pos_ in ('NOUN', 'PROPN')])
    )
    iterator = captions.reindex(
        np.random.permutation(captions.index)
    ).iterrows()
    for idx, row in tqdm(iterator, total=len(captions),
                         leave=False, desc="Captions"):
        fname = img_dir / Path(row.fname)
        if fname.exists():
            for noun in nouns_s[idx]:
                yield str(fname), noun


def generate_batches(captions, img_dir, batch_size=1):
    _, get_image_features.cnn_model = create_cnn_model()

    i = 0
    X = np.zeros((batch_size, IMG_OUT_SHAPE + WORD_OUT_SHAPE), dtype='float32')
    fnames = [None] * batch_size
    nouns = list(iter_nouns(captions, img_dir))
    for fname, noun in tqdm(nouns, desc="fname-noun"):
        img_features = get_image_features(fname)
        X[i] = np.r_[img_features[0], noun.vector]
        fnames[i] = fname
        i += 1
        if i == batch_size:
            i = 0
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            X[np.isnan(X)] = 0
            yield X, fnames
            X = np.zeros_like(X)
            fnames = [None] * batch_size


@click.command()
@click.option('--batch_size', default=10)
@click.option('--img_dir', type=click.Path(exists=True, writable=True,
                                           file_okay=False),
              default='/data/master_thesis/Flicker8k_Dataset')
@click.option('--tokens_file', type=click.Path(exists=True, dir_okay=False),
              default='/data/master_thesis/Flickr8k.lemma.token.txt')
@click.option('--save_to_dir', type=click.Path(exists=True, writable=True,
                                               file_okay=False),
              default=f'/data/master_thesis/{__file__.parent.name}')
def main(batch_size, img_dir, tokens_file, save_to_dir):
    click.secho(f"batch_size: {batch_size}\n"
                f"img_dir: {img_dir}\n"
                f"tokens_file: {tokens_file}\n"
                f"save_to_dir: {save_to_dir}\n",
                color="blue")
    save_to_dir = Path(save_to_dir)
    captions, fnames = load_captions(tokens_file)
    train_fnames, test_fnames = train_test_split(fnames, test_size=1000)
    train_captions = captions.loc[captions.fname.isin(train_fnames)]
    test_captions = captions.loc[captions.fname.isin(test_fnames)]
    batch_generator = generate_batches(
        captions=train_captions,
        img_dir=img_dir, batch_size=batch_size
    )
    for i, (batch, fnames) in enumerate(batch_generator):
        np.save(save_to_dir / f'features_som_{i}.npy',
                {'batch': batch, 'fnames': fnames})
    train_captions.to_csv(save_to_dir / 'train_captions.csv', index=False)
    test_captions.to_csv(save_to_dir / 'test_captions.csv', index=False)


if __name__ == '__main__':
    main()
