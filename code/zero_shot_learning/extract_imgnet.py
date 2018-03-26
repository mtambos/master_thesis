from pathlib import Path

import tarfile

from tqdm import tqdm


for fname in tqdm(sorted(Path('./').glob('n*.tar'))):
    fdir = fname.parent / fname.stem
    if not fdir.exists():
        fdir.mkdir()
    tf = tarfile.open(fname)
    tf.extractall(fdir)
