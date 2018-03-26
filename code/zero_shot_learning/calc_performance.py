#!/usr/bin/env python
import json
from pathlib import Path
import pickle

import click
from multiset import Multiset
from statsrecorder import StatsRecorder
from tqdm import tqdm


def hp_at_k(y_true, y_pred, correct_sets, k):
    return len(Multiset(y_pred[:k]) & Multiset(correct_sets[y_true]))


def load_results(fname):
    with fname.open('rb') as fp:
        results = pickle.load(fp)
    return results


def load_correct_sets(fname):
    with fname.open() as fp:
        correct_sets = json.load(fp)
    return correct_sets


@click.command()
@click.option('--results_dir', type=click.Path(exists=True, file_okay=False),
              default=f'/data/master_thesis/zero_shot_learning')
@click.option('--correct_sets_file', type=click.Path(exists=True, dir_okay=False),
              default=f'./imgnet_correct_sets.json')
@click.option('--file_pattern', default='**/*_predictions.pickle')
@click.option('--output_file', type=click.Path(dir_okay=False),
              default=f'./zero_shot_performance.json')
def main(results_dir, correct_sets_file, file_pattern, output_file):
    results = {}
    results_dir = Path(results_dir)
    correct_sets_file = Path(correct_sets_file)
    output_file = Path(output_file)

    click.secho(
        f"results_dir: {results_dir}\n"
        f"correct_sets_file: {correct_sets_file}\n"
        f"file_pattern: {file_pattern}\n"
        f"output_file: {output_file}\n"
    )

    correct_sets = load_correct_sets(correct_sets_file)
    if '1k_2hop' in correct_sets:
        correct_sets['1k_2hops'] = correct_sets['1k_2hop']
        del correct_sets['1k_2hop']

    if '1k_3hop' in correct_sets:
        correct_sets['1k_3hops'] = correct_sets['1k_3hop']
        del correct_sets['1k_3hop']

    ks = [1, 2, 5, 10, 20]
    dataset_names = ['1k', '1k_2hops', '1k_3hops', '21k']
    hrc_correct_at_k = {ds_name: {k: [StatsRecorder(), 0, 0] for k in ks}
                        for ds_name in dataset_names}
    flt_correct_at_k = {ds_name: {k: StatsRecorder() for k in ks}
                        for ds_name in dataset_names}

    click.secho('Listing result files')
    result_files = list(results_dir.glob(file_pattern))
    for fname in tqdm(result_files, desc='Files'):
        results = load_results(fname)
        for ds_name, correct_at_k_ds in tqdm(hrc_correct_at_k.items(),
                                             total=len(hrc_correct_at_k),
                                             leave=False,
                                             desc='Datasets'):
            correct_sets_ds = correct_sets[ds_name]
            for key, y_pred in tqdm(results[f'results_{ds_name}'].items(),
                                    total=len(results[f'results_{ds_name}']),
                                    leave=False,
                                    desc='Images'):
                for k in ks:
                    correct_sets_ds_at_k = correct_sets_ds[str(k)]
                    y_true = key[0]
                    if y_true in correct_sets_ds_at_k:
                        correct_at_k_ds[k][0].update(hp_at_k(y_true, y_pred, correct_sets_ds_at_k, k))
                        flt_correct_at_k[ds_name][k].update(1 if y_true in y_pred[:k] else 0)
                    correct_at_k_ds[k][1] = len(results[f'results_{ds_name}'])
                    correct_at_k_ds[k][2] += len(correct_sets[ds_name][str(k)])

    hrc_correct_at_k = {ds_name: {k: (hrc_correct_at_k[ds_name][k][0].mean[0],
                                      hrc_correct_at_k[ds_name][k][0].nobservations,
                                      hrc_correct_at_k[ds_name][k][1],
                                      hrc_correct_at_k[ds_name][k][2])
                                  for k in ks}
                        for ds_name in dataset_names}
    flt_correct_at_k = {ds_name: {k: (flt_correct_at_k[ds_name][k].mean[0],
                                      flt_correct_at_k[ds_name][k].nobservations)
                                  for k in ks}
                        for ds_name in dataset_names}

    with output_file.open('w') as fp:
        json.dump({'hierarchical': hrc_correct_at_k, 'flat': flt_correct_at_k}, fp)


if __name__ == '__main__':
    main()
