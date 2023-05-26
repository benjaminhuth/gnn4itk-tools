#!/usr/bin/python3

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from tqdm import tqdm


def collect_data(feature_store_dir, train_all_dir, max_events):
    raw_hit_list, meas_hit_list, reader_hit_list, track_hit_list, target_hit_list = (
        [],
        [],
        [],
        [],
        [],
    )

    files = list(feature_store_dir.glob("*-graph.pyg"))

    for torch_file in tqdm(files[: min(len(files), max_events)]):
        torch_data = torch.load(torch_file, map_location="cpu")
        event_id = torch_file.name[:14]

        raw_hits = pd.read_csv(train_all_dir / f"{event_id}-hits.csv")
        raw_hit_list.append(len(raw_hits))

        meas_idxs = pd.read_csv(
            train_all_dir / f"{event_id}-measurement-simhit-map.csv"
        ).hit_id
        meas_hit_list.append(len(raw_hits.loc[meas_idxs, :]))

        reader_hit_list.append(len(torch_data.r))
        track_hit_list.append(len(np.unique(torch_data.track_edges.flatten())))
        target_hit_list.append(sum((torch_data.pt > 0.5) & (torch_data.nhits >= 3)))

    return raw_hit_list, meas_hit_list, reader_hit_list, track_hit_list, target_hit_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("feature_store_dir", type=str)
    parser.add_argument("train_all_dir", type=str)
    parser.add_argument("--max_events", "-n", type=int, default=sys.maxsize)

    args = vars(parser.parse_args())

    res = collect_data(
        Path(args["feature_store_dir"]), Path(args["train_all_dir"]), args["max_events"]
    )

    labels = ["sim hits", "meas hits", "reader hits", "track hits", "target hits"]
    max_label_length = max([len(l) for l in labels])

    vals = []
    errs = []
    strs = []

    for data in res:
        vals.append(np.mean(data))
        errs.append(np.std(data))
        strs.append("{:.1f}K +- {:.1f}K".format(vals[-1] / 1000, errs[-1] / 1000))

    max_str_length = max([len(s) for s in strs])

    cols = os.get_terminal_size().columns - (max_str_length + max_label_length + 7)

    print()

    for label, val, err, val_err_str in zip(labels, vals, errs, strs):
        n_cols = round(cols * (val / max(vals)))

        print(
            "{} | {} | {}".format(
                label.ljust(max_label_length),
                val_err_str.ljust(max_str_length),
                n_cols * "█",
            )
        )
