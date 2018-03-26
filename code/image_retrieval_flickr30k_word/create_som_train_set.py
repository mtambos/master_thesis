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
    captions = pd.read_table(tokens_file, names=['sentence'])['sentence']
    return captions.str.replace(r'\[/EN#\d+(/\w+)+ ([^]]+)\]', r'\2')


def get_word_embeddings(img_fname, tokens_dir, nlp):
    tokens_file = tokens_dir / img_fname.name
    tokens_file = tokens_file.with_suffix('.txt')
    captions = load_captions(tokens_file)
    nlp_words = [(i, word) for i, caption in enumerate(captions)
                 for word in nlp(caption)
                 if word.pos_ in ('NOUN', 'PROPN')]
    nlp_words = [np.r_[word.vector / word.vector_norm, (i, str(word))]
                 for i, word in nlp_words]
    nlp_words = pd.DataFrame(
        nlp_words,
        columns=[f'vector_{i}' for i in range(300)] + ['caption_id', 'word']
    )
    nlp_words['caption_id'] = nlp_words.caption_id.astype(int)
    nlp_words = nlp_words.drop_duplicates(subset=['word'])
    nlp_words['fname'] = img_fname
    return nlp_words


def generate_words_df(words_fname, img_dir, tokens_dir, nlp):
    words_fname = Path(words_fname)
    img_dir = Path(img_dir)
    tokens_dir = Path(tokens_dir)
    if words_fname.exists():
        words = pd.read_csv(words_fname)
    else:
        img_fnames = list(img_dir.glob('*.jpg'))
        words = [get_word_embeddings(fname, tokens_dir, nlp)
                 for fname in img_fnames]
        words = pd.concat(words, ignore_index=True)
        words.to_csv(words_fname, index=False)
    return words


def generate_batches(captions, img_dir, batch_size=1):
    captions = captions.sample(frac=1, replace=False)
    vec_cols = captions.columns[captions.columns.str.match(r'vector_\d+')]

    _, get_image_features.cnn_model = create_cnn_model()
    i = 0
    X = np.zeros((batch_size, IMG_OUT_SHAPE + WORD_OUT_SHAPE), dtype='float32')
    batch_fnames = [None] * batch_size
    for _, row in tqdm(captions.iterrows(), total=len(captions),
                       desc="fname-noun"):
        fname = row.fname
        img_features = get_image_features(fname)
        caption_vector = row.loc[vec_cols]
        X[i] = np.r_[img_features[0], caption_vector]
        batch_fnames[i] = fname
        i += 1
        if i == batch_size:
            i = 0
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            X[np.isnan(X)] = 0
            yield X, batch_fnames
            X = np.zeros_like(X)
            batch_fnames = [None] * batch_size


@click.command()
@click.option('--batch_size', default=10)
@click.option('--img_dir', type=click.Path(exists=True, writable=True,
                                           file_okay=False),
              default='/data/master_thesis/Flickr30k_Dataset')
@click.option('--tokens_dir', type=click.Path(exists=True, file_okay=False),
              default='/data/master_thesis/Flickr30kEntities/Sentences/')
@click.option('--save_to_dir', type=click.Path(exists=True, writable=True,
                                               file_okay=False),
              default=f'/data/master_thesis/{__file__.parent.name}')
def main(batch_size, img_dir, tokens_dir, save_to_dir):
    click.secho(f"batch_size: {batch_size}\n"
                f"img_dir: {img_dir}\n"
                f"tokens_dir: {tokens_dir}\n"
                f"save_to_dir: {save_to_dir}\n",
                color="blue")
    save_to_dir = Path(save_to_dir)
    tokens_dir = Path(tokens_dir)
    nlp = spacy.load('en_core_web_md')
    captions = generate_words_df(save_to_dir / 'words.csv',
                                 img_dir, tokens_dir, nlp)
    fnames = np.unique(captions.fname)
    train_fnames, test_fnames = train_test_split(fnames, test_size=1000)
    train_captions = captions[captions.fname.isin(train_fnames)]
    test_captions = captions[captions.fname.isin(test_fnames)]

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
