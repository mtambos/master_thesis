# pylint: disable=too-many-arguments,R0914,R0915,C0111,C0103
import os
from pathlib import Path

import click
import pandas as pd
import spacy
from tqdm import tqdm


@click.command()
@click.option('--imgs_dir', type=click.Path(exists=True, file_okay=False),
              default=f'/data/master_thesis/ILSVRC2012_img_train')
@click.option('--save_to_dir', type=click.Path(exists=True, file_okay=False),
              default=f'/data/master_thesis/zero_shot_learning')
@click.option('--results_prefix', default=f'cnn_features')
@click.option('--batch_size', type=int, default=1024)
@click.option('--cuda_device_id', type=str, default='3')
def main(imgs_dir, save_to_dir, results_prefix, batch_size, cuda_device_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_id
    from multimodal_som import ImgIterator, create_cnn_model

    imgs_dir = Path(imgs_dir)
    save_to_dir = Path(save_to_dir)

    click.secho(
        f"imgs_dir: {imgs_dir}\n"
        f"save_to_dir: {save_to_dir}\n"
        f"results_prefix: {results_prefix}\n"
        f"batch_size: {batch_size}\n"
        f"cuda_device_id: {cuda_device_id}\n"
    )

    _, cnn_model = create_cnn_model()
    nlp = spacy.load('en_core_web_md')
    iterator = ImgIterator(imgs_dir=imgs_dir, batch_size=batch_size, epochs=1,
                           cnn_model=cnn_model, nlp=nlp)
    iterator = enumerate(tqdm(iterator, total=iterator.total_batches))

    col_names = [f'vec_{i}' for i in range(4096)]
    for i, (batch, labels) in iterator:
        try:
            data = pd.DataFrame(batch, columns=col_names)
            data['label'] = labels
            data.to_csv(save_to_dir / f'{results_prefix}_{i}.csv')
        except Exception as e:  # pylint: disable=W0703
            print(f'In main loop: {e}')


if __name__ == '__main__':
    main()  # pylint: disable=E1120
