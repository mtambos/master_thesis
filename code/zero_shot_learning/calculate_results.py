# pylint: disable=too-many-arguments,R0914,R0915,C0111,C0103
import json
from pathlib import Path
import pickle
import os
import uuid

import arrow
import click
from decorator import decorate
from joblib import delayed, Parallel
import networkx as nx
from networkx.algorithms.shortest_paths.generic import (
    shortest_path_length as nx_shortest_path_length
)
from networkx.algorithms.shortest_paths.unweighted import (
    single_source_shortest_path as nx_single_source_shortest_path
)
import pandas as pd
import psutil
import spacy
from tqdm import tqdm, trange

try:
    from multimodal_som import MultimodalSOM
except ImportError:
    from .multimodal_som import MultimodalSOM


def get_nhop_set(base_wnids, wn_is_a, n_hops, super_wnids=None, tqdm_f=tqdm):
    graph = nx.Graph()
    graph.add_edges_from(wn_is_a.as_matrix())
    base_wnids = set(base_wnids)
    if super_wnids is None:
        super_wnids = base_wnids
    else:
        super_wnids = set(super_wnids)
        super_wnids -= base_wnids

    ret_val = set(base_wnids)
    for base_wnid in tqdm_f(base_wnids):
        paths = nx_shortest_path_length(graph, source=base_wnid)
        paths = sorted(paths.items(), key=lambda x: x[1])
        for wnid, hops in paths:
            if hops > n_hops:
                break
            elif wnid in super_wnids:
                ret_val.add(wnid)
    return ret_val


def get_nhop_k_set(true_wnids, wn_is_a, k, valid_f=None, tqdm_f=tqdm):
    graph = nx.Graph()
    graph.add_edges_from(wn_is_a.as_matrix())
    true_wnids = set(true_wnids)

    ret_val = {true_wnid: set() for true_wnid in true_wnids}
    for true_wnid in tqdm_f(true_wnids):
        correct_true_wnid = ret_val[true_wnid]
        if true_wnid not in graph.nodes:
            correct_true_wnid.add(true_wnid)
            continue
        paths = nx_shortest_path_length(graph, source=true_wnid)
        paths = sorted(paths.items(), key=lambda x: x[1])
        n_hops = 0
        while len(correct_true_wnid) < k:
            radius_set = {true_wnid}
            for wnid, hops in paths:
                if hops > n_hops:
                    break
                radius_set.add(wnid)
            if valid_f is not None:
                radius_set = valid_f(radius_set, graph)
            correct_true_wnid.update(radius_set)
            n_hops += 1
    return ret_val


def valid_sset(wnids_set, graph, base_wnids, max_hops):
    ret_val = {
        wnid for wnid in wnids_set
        if (nx_single_source_shortest_path(graph, wnid, cutoff=max_hops) &
            base_wnids)
    }
    return ret_val


def create_vec2wnid(vec2wnid_fname, wnid2words_fname, nlp):
    vec2wnid_fname = Path(vec2wnid_fname)
    wnid2words_fname = Path(wnid2words_fname)
    if vec2wnid_fname.suffix.lower() == '.csv':
        if vec2wnid_fname.exists():
            vec2wnid = pd.read_csv(vec2wnid_fname)
        else:
            wnid2word = pd.read_table(
                wnid2words_fname, na_values=[''], names=['wnid', 'words'],
                index_col='wnid', dtype={'wnid': str, 'words': str}
            )
            vec2wnid = []
            for wnid, row in tqdm(wnid2word.iterrows(), total=len(wnid2word)):
                for word in str(row.words).split(','):
                    vec2wnid.append(
                        (*(nlp(word).vector/nlp(word).vector_norm), word, wnid)
                    )
            vec2wnid = pd.DataFrame(
                vec2wnid,
                columns=[f'vec_{i}' for i in range(300)]+['word', 'wnid']
            )
            vec2wnid.to_csv('vec2wnid.csv', index=False)
    elif vec2wnid_fname.suffix.lower() == '.pickle':
        with open(vec2wnid_fname, 'rb') as fp:
            vec2wnid = pickle.load(fp)
    else:
        raise ValueError(f"Suffix '{vec2wnid_fname.suffix}' not supported.")

    return vec2wnid


def load_imgnet_correct_wnids(imgnet_correct_wnids):
    with open(imgnet_correct_wnids, 'r') as fp:  # pylint: disable=C0103
        imgnet_correct_wnids = set(json.load(fp))
    return imgnet_correct_wnids


def get_vec2wnid_sets(onek_wnid_file, vec2wnid_file,
                      onek_2hops_wnid_file, onek_3hops_wnid_file,
                      wnid_map_file):
    nlp = spacy.load('en_core_web_md')

    onek_wnid = load_imgnet_correct_wnids(onek_wnid_file)
    onek_2hops_wnid = load_imgnet_correct_wnids(onek_2hops_wnid_file)
    onek_3hops_wnid = load_imgnet_correct_wnids(onek_3hops_wnid_file)

    vec2wnid_21k = create_vec2wnid(vec2wnid_file, wnid_map_file, nlp)
    vec2wnid_20k = vec2wnid_21k.loc[~vec2wnid_21k.wnid.isin(onek_wnid)]
    vec2wnid_1k = vec2wnid_21k.loc[vec2wnid_21k.wnid.isin(onek_wnid)]
    vec2wnid_1k_2hops = vec2wnid_21k.loc[
        vec2wnid_21k.wnid.isin(onek_2hops_wnid)
    ]
    vec2wnid_1k_3hops = vec2wnid_21k.loc[
        vec2wnid_21k.wnid.isin(onek_3hops_wnid)
    ]
    vec_cols = vec2wnid_21k.columns[
        vec2wnid_21k.columns.str.startswith('vec')
    ]
    vec2wnid_21k_vecs = vec2wnid_21k.loc[:, vec_cols]
    vec2wnid_20k_vecs = vec2wnid_20k.loc[:, vec_cols]
    vec2wnid_1k_vecs = vec2wnid_1k.loc[:, vec_cols]
    vec2wnid_1k_2hops_vecs = vec2wnid_1k_2hops.loc[:, vec_cols]
    vec2wnid_1k_3hops_vecs = vec2wnid_1k_3hops.loc[:, vec_cols]

    return (vec2wnid_21k, vec2wnid_20k, vec2wnid_1k,
            vec2wnid_1k_2hops, vec2wnid_1k_3hops,
            vec2wnid_21k_vecs, vec2wnid_20k_vecs,
            vec2wnid_1k_vecs, vec2wnid_1k_2hops_vecs, vec2wnid_1k_3hops_vecs)


def _timeit(func, *args, **kwargs):
    pre_time = arrow.now()
    result = func(*args, **kwargs)
    post_time = arrow.now()
    fid = uuid.uuid4()
    pid = os.getpid()
    ps_proc = psutil.Process(pid)
    memory_use = ps_proc.memory_info().rss / (2**20)
    print(
        f'{arrow.now().isoformat()}'
        f'{func.__name__}({fid}): '
        f'{(post_time - pre_time).total_seconds():0.2f}s / '
        f'{memory_use} MB'
    )
    return result


def timeit(func):
    return decorate(func, _timeit)


@timeit
def timed_load_csv(fname):
    if fname.suffix.lower() == '.csv':
        return pd.read_csv(fname).drop('Unnamed: 0', axis=1)
    elif fname.suffix.lower() == '.pickle':
        with open(fname, 'rb') as fp:
            return pickle.load(fp)
    else:
        raise ValueError(f"Suffix '{fname.suffix}' not supported.")


@timeit
def timed_get_wnids(mm_som, images_features, vec2wnid_vecs, vec2wnid,
                    labels, i):
    knn_ids_21k = mm_som.get_knn_words(
        images_features=images_features, k=40, vec2word=vec2wnid_vecs
    )
    wnid_sets = vec2wnid.wnid.iloc[knn_ids_21k.flatten()].values
    wnid_sets = wnid_sets.reshape(knn_ids_21k.shape)
    return {(label, i, j): wnid_set
            for j, (label, wnid_set) in enumerate(zip(labels, wnid_sets))}


def process_file(fname, results_file, batch_size, i, model_file_prefix,
                 onek_wnid_file, vec2wnid_file, onek_2hops_wnid_file,
                 onek_3hops_wnid_file, wnid_map_file, skip_processed):
    fname = Path(fname)
    results_file = Path(results_file)
    if skip_processed and results_file.exists():
        try:
            with open(results_file, 'rb') as fp:
                pickle.load(fp)
            return
        except pickle.UnpicklingError:
            pass

    onek_wnid_file = Path(onek_wnid_file)
    vec2wnid_file = Path(vec2wnid_file)
    onek_2hops_wnid_file = Path(onek_2hops_wnid_file)
    onek_3hops_wnid_file = Path(onek_3hops_wnid_file)
    wnid_map_file = Path(wnid_map_file)

    vec2wnid_sets = get_vec2wnid_sets(
        onek_wnid_file, vec2wnid_file, onek_2hops_wnid_file,
        onek_3hops_wnid_file, wnid_map_file
    )
    (vec2wnid_21k, _, vec2wnid_1k,
     vec2wnid_1k_2hops, vec2wnid_1k_3hops,
     vec2wnid_21k_vecs, _,
     vec2wnid_1k_vecs, vec2wnid_1k_2hops_vecs,
     vec2wnid_1k_3hops_vecs) = vec2wnid_sets

    mm_som = MultimodalSOM(rows=100, cols=100, epochs=1, load_cnn=False)
    mm_som.load(model_file_prefix)
    results_21k = {}
    results_1k = {}
    results_1k_2hops = {}
    results_1k_3hops = {}
    data = timed_load_csv(fname)
    for j in trange(0, len(data), batch_size, desc='Batches'):
        batch = data.iloc[j: j + batch_size]
        labels = batch.label
        images_features = batch.iloc[:, :-1]

        results_21k.update(timed_get_wnids(
            mm_som=mm_som, images_features=images_features,
            vec2wnid_vecs=vec2wnid_21k_vecs, vec2wnid=vec2wnid_21k,
            labels=labels, i=i
        ))

        results_1k.update(timed_get_wnids(
            mm_som=mm_som, images_features=images_features,
            vec2wnid_vecs=vec2wnid_1k_vecs, vec2wnid=vec2wnid_1k,
            labels=labels, i=i
        ))

        results_1k_2hops.update(timed_get_wnids(
            mm_som=mm_som, images_features=images_features,
            vec2wnid_vecs=vec2wnid_1k_2hops_vecs,
            vec2wnid=vec2wnid_1k_2hops, labels=labels, i=i
        ))

        results_1k_3hops.update(timed_get_wnids(
            mm_som=mm_som, images_features=images_features,
            vec2wnid_vecs=vec2wnid_1k_3hops_vecs,
            vec2wnid=vec2wnid_1k_3hops, labels=labels, i=i
        ))

    fname_results_file = (
        results_file.parent /
        f'{fname.stem}_{results_file.stem}{results_file.suffix}'
    )
    with open(fname_results_file, 'wb') as fp:
        pickle.dump(
            {
                'results_21k': results_21k,
                'results_1k': results_1k,
                'results_1k_2hops': results_1k_2hops,
                'results_1k_3hops': results_1k_3hops,
            },
            fp
        )

    return


@click.command()
@click.option('--data_dir', type=click.Path(exists=True, file_okay=False),
              default=f'/data/master_thesis/zero_shot_learning')
@click.option('--wnid_map_file', type=click.Path(exists=True, dir_okay=False),
              default=f'/data/master_thesis/wnid2words.txt')
@click.option('--onek_wnid_file', type=click.Path(exists=True, dir_okay=False),
              default=f'./imgnet_1k_wnids.json')
@click.option('--onek_2hops_wnid_file',
              type=click.Path(exists=True, dir_okay=False),
              default=f'./imgnet_1k_2hops_wnids.json')
@click.option('--onek_3hops_wnid_file',
              type=click.Path(exists=True, dir_okay=False),
              default=f'./imgnet_1k_3hops_wnids.json')
@click.option('--vec2wnid_file', type=click.Path(exists=True, dir_okay=False),
              default=f'./vec2wnid.csv')
@click.option('--results_file', type=click.Path(dir_okay=False, writable=True,
                                                resolve_path=True),
              default=f'./results_words.pickle')
@click.option('--model_file_prefix', default='som_planar_rectangular')
@click.option('--batch_size', default=20)
@click.option('--file_pattern', default='**/*.[cC][sS][vV]')
@click.option('--n_jobs', default=-1, type=int)
@click.option('--skip_processed', default=False, type=bool)
def main(data_dir, wnid_map_file, onek_wnid_file, onek_2hops_wnid_file,
         onek_3hops_wnid_file, vec2wnid_file, results_file,
         model_file_prefix, batch_size, file_pattern, n_jobs,
         skip_processed):
    data_dir: Path = Path(data_dir).resolve()
    wnid_map_file = str(Path(wnid_map_file).resolve())
    onek_wnid_file = str(Path(onek_wnid_file).resolve())
    onek_2hops_wnid_file = str(Path(onek_2hops_wnid_file).resolve())
    onek_3hops_wnid_file = str(Path(onek_3hops_wnid_file).resolve())
    vec2wnid_file = str(Path(vec2wnid_file).resolve())
    results_file = str(Path(results_file).resolve())

    click.secho(
        f"data_dir: {data_dir}\n"
        f"wnid_map_file: {wnid_map_file}\n"
        f"onek_wnid_file: {onek_wnid_file}\n"
        f"onek_2hops_wnid_file: {onek_2hops_wnid_file}\n"
        f"onek_3hops_wnid_file: {onek_3hops_wnid_file}\n"
        f"vec2wnid_file: {vec2wnid_file}\n"
        f"results_file: {results_file}\n"
        f"model_file_prefix: {model_file_prefix}\n"
        f"batch_size: {batch_size}\n"
        f"file_pattern: {file_pattern}\n"
        f"n_jobs: {n_jobs}\n"
        f"skip_processed: {skip_processed}\n"
    )

    # pylint: disable=E1101
    iterator = list(data_dir.glob(file_pattern))
    iterator = enumerate(tqdm(iterator, desc='Files'))

    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(
            str(fname), results_file, batch_size, i, model_file_prefix,
            onek_wnid_file, vec2wnid_file, onek_2hops_wnid_file,
            onek_3hops_wnid_file, wnid_map_file, skip_processed
        )
        for i, fname in iterator
    )


if __name__ == '__main__':
    main()  # pylint: disable=E1120
