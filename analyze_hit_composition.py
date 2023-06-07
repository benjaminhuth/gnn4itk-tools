#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from tqdm import tqdm

from gnn4itk_tools.utils import collect_hit_distribution_data as collect_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("feature_store_dir", type=str)
    parser.add_argument("train_all_dir", type=str)
    parser.add_argument("--max_events", "-n", type=int, default=sys.maxsize)

    args = vars(parser.parse_args())

    res = collect_data(
        Path(args["feature_store_dir"]), Path(args["train_all_dir"]), args["max_events"]
    )

    labels = res.keys()
    counts = [res[key] for key in labels]

    max_label_length = max([len(l) for l in labels])

    vals = []
    errs = []
    strs = []

    for count in counts:
        vals.append(np.mean(count))
        errs.append(np.std(count))
        strs.append("{:.1f}K +- {:.1f}K".format(vals[-1] / 1000, errs[-1] / 1000))

    max_str_length = max([len(s) for s in strs])

    cols = os.get_terminal_size().columns - (max_str_length + max_label_length + 7)

    print(cols)

    for label, val, err, val_err_str in zip(labels, vals, errs, strs):
        n_cols = round(cols * (val / max(vals)))

        print(
            "{} | {} | {}".format(
                label.ljust(max_label_length),
                val_err_str.ljust(max_str_length),
                n_cols * "█",
            )
        )
