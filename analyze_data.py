#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
from multiprocessing import Pool
from functools import partial

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


def cantor_pairing(a):
    a = np.sort(a, axis=0)
    return a[1] + ((a[0] + a[1]) * (a[0] + a[1] + 1)) // 2


def effpur(true, pred):
    cantor_true = cantor_pairing(true)
    cantor_pred = cantor_pairing(pred)
    cantor_intersection = np.intersect1d(cantor_true, cantor_pred)

    return {
        "eff": len(cantor_intersection)
        / len(cantor_true),  # if len(cantor_true) > 0 else 0,
        "pur": len(cantor_intersection)
        / len(cantor_pred),  # if len(cantor_pred) > 0 else 0,
    }


def df_from_dict(d):
    return pd.DataFrame({key: [value] for key, value in d.items()})


def process(filename, cut, target_pt, target_nhits):
    try:
        event = torch.load(filename, map_location="cpu")
    except:
        print("Failed to load", filename)
        return pd.DataFrame()

    res = {}

    true_edges = event.track_edges
    target_true_edges = true_edges[
        :, torch.logical_and(event.pt > target_pt, event.nhits > target_nhits)
    ]

    res["num_hits"] = len(event.r)
    res["num_true_edges"] = event.track_edges.shape[1]

    res["target_share"] = target_true_edges.shape[1] / true_edges.shape[1]
    res["noise_share"] = 1.0 - (
        len(np.unique(event.track_edges.flatten())) / len(event.r)
    )

    if target_true_edges.shape[1] == 0:
        res["zero_target_tracks"] = True
    else:
        res["zero_target_tracks"] = False

    if not "edge_index" in event:
        return df_from_dict(res)

    if "scores" in event:
        edge_index = event.edge_index[:, event.scores > cut]
    else:
        edge_index = event.edge_index

    metrics_all = effpur(true_edges, edge_index)
    res["eff"] = metrics_all["eff"]
    res["pur"] = metrics_all["pur"]

    if not res["zero_target_tracks"]:
        metrics_target = effpur(target_true_edges, edge_index)
        res["target_eff"] = metrics_target["eff"]
        res["target_pur"] = metrics_target["pur"]

    return df_from_dict(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument("path", type=str)
    parser.add_argument("--cut", "-c", type=float, default=0.5)
    parser.add_argument("--max", type=int)
    parser.add_argument("--jobs", "-j", type=int, default=16)
    args = vars(parser.parse_args())

    path = Path(args["path"])
    assert path.exists()
    assert args["cut"] > 0 and args["cut"] < 1

    target_pt = 0.5
    target_nhits = 3
    print(
        "score_cut:", args["cut"], "target_pt", target_pt, "target_nhits", target_nhits
    )

    func = partial(
        process, cut=args["cut"], target_pt=target_pt, target_nhits=target_nhits
    )
    filenames = list(path.glob("*.pyg"))
    filenames = filenames[: args["max"] if "max" in args else len(filenames)]
    assert len(filenames) > 0

    if args["jobs"] > 0:
        with Pool(min(args["jobs"], len(filenames))) as p:
            res = list(tqdm(p.imap(func, filenames), total=len(filenames)))
    else:
        res = [func(filename) for filename in tqdm(filenames)]

    res = pd.concat(res)

    print("Events with 0 target tracks:", sum(res["zero_target_tracks"]))

    l = max([len(c) for c in res.columns])
    for c in res.columns:
        if res.dtypes[c] == bool:
            continue

        vals = res[c]
        vals = vals[~pd.isnull(vals)].to_numpy()
        print(
            c.ljust(l),
            "avg: {:.3f} +- {:.3f}, max: {:.3f}, min: {:.3f}".format(
                np.mean(vals), np.std(vals), max(vals), min(vals)
            ),
        )
