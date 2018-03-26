from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from tqdm import tqdm

try:
    from .multimodal_som import get_image_features, create_cnn_model
except ImportError:
    # noinspection PyUnresolvedReferences
    from multimodal_som import get_image_features, create_cnn_model


IMG_IN_SHAPE = (3, 227, 227)
IMG_OUT_SHAPE = 4096
WORD_OUT_SHAPE = 300
try:
    # noinspection PyUnboundLocalVariable
    __file__ = Path(__file__).resolve()
except:
    __file__ = Path('.').resolve()


def load_captions(tokens_file, img_dir):
    captions = pd.read_table(tokens_file, names=['fname', 'doc'])
    captions['fname'] = captions.fname.str.replace(r'[#.]\d', '')
    fnames = set([f.name for f in img_dir.glob('*.jpg')])
    captions = captions.loc[captions.fname.isin(fnames)]
    return captions, fnames


def get_sentence_embeddings(tokens_file, img_dir, nlp):
    captions, _ = load_captions(tokens_file, img_dir)
    nlp_captions = [nlp(caption) for caption in captions.doc]
    nlp_captions = [np.r_[caption.vector / caption.vector_norm, i]
                    for i, caption in enumerate(nlp_captions)]
    nlp_captions = pd.DataFrame(
        nlp_captions,
        columns=[f'vector_{i}' for i in range(300)] + ['caption_id']
    )
    nlp_captions['caption_id'] = nlp_captions.caption_id.astype(int)
    nlp_captions['caption'] = captions.doc
    nlp_captions['fname'] = captions.fname
    return nlp_captions


def generate_sentences_df(sentences_fname, img_dir, tokens_file):
    sentences_fname = Path(sentences_fname)
    tokens_file = Path(tokens_file)
    if sentences_fname.exists():
        sentences = pd.read_csv(sentences_fname)
    else:
        nlp = spacy.load('en_core_web_md')
        sentences = get_sentence_embeddings(tokens_file, img_dir, nlp)
        sentences.to_csv(sentences_fname, index=False)
    return sentences


# noinspection PyPep8Naming
def generate_batches(captions, img_dir, batch_size=1):
    captions = captions.sample(frac=1, replace=False)
    vec_cols = captions.columns[captions.columns.str.match(r'vector_\d+')]

    _, get_image_features.cnn_model = create_cnn_model()
    i = 0
    X = np.zeros((batch_size, IMG_OUT_SHAPE + WORD_OUT_SHAPE), dtype='float32')
    batch_fnames = [None] * batch_size
    for _, row in tqdm(captions.iterrows(), total=len(captions),
                       desc="fname-noun"):
        fname = img_dir / row.fname
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
              default='/data/master_thesis/Flickr8k_Dataset')
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
    img_dir = Path(img_dir)
    tokens_file = Path(tokens_file)
    save_to_dir = Path(save_to_dir)
    captions = generate_sentences_df(save_to_dir / 'sentences.csv',
                                     img_dir, tokens_file)
    fnames = captions.fname.value_counts().index
    train_fnames, test_fnames = train_test_split(fnames, test_size=1000)
    train_captions = captions.loc[captions.fname.isin(train_fnames)]
    test_captions = captions.loc[captions.fname.isin(test_fnames)]
    batch_generator = generate_batches(
        captions=train_captions,
        img_dir=img_dir, batch_size=batch_size
    )
    for i, (batch, fnames) in enumerate(batch_generator):
        # noinspection PyTypeChecker
        np.save(save_to_dir / f'features_som_{i}.npy',
                {'batch': batch, 'fnames': fnames})
    train_captions.to_csv(save_to_dir / 'train_captions.csv', index=False)
    test_captions.to_csv(save_to_dir / 'test_captions.csv', index=False)


if __name__ == '__main__':
    main()
