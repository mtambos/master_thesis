from pathlib import Path

import numpy as np
from tqdm import tqdm

from multimodal_som import MultimodalSOM

__file_path__ = Path(__file__).resolve()


def main():
    IMG_OUT_SHAPE = 4096
    WORD_OUT_SHAPE = 300
    FEATURE_OUT_SHAPE = IMG_OUT_SHAPE + WORD_OUT_SHAPE

    msom_model = MultimodalSOM()
    msom_model.load('som_planar_rectangular')

    save_to_dir = Path(f'/data/master_thesis/{__file_path__.parent.name}')

    som_feature_files = list(set(
        save_to_dir.glob(f'**/features_som_*.npy')
    ))
    print(save_to_dir)
    print(len(som_feature_files))
    for i, feature_path in enumerate(tqdm(som_feature_files)):
        data = np.load(feature_path)
        som_features = data[:, :FEATURE_OUT_SHAPE]
        class_ids = data[:, FEATURE_OUT_SHAPE:]
        top_features = som_features.copy()
        for j, som_features in enumerate(som_features):
            target = top_features[j, :IMG_OUT_SHAPE].copy()
            bmu = msom_model.get_bmu(img_target=target, k=1, metric='sqeuclidean')[0]
            bmu = msom_model.som_model.codebook[bmu]
            top_features[j, IMG_OUT_SHAPE:] = bmu[IMG_OUT_SHAPE:]
        np.save(save_to_dir / f'features_top_{i}.npy',
                np.r_['1,2', top_features, class_ids])


if __name__ == '__main__':
    main()
