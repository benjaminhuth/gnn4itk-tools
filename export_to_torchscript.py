#!/usr/bin/env python3

import argparse
from pathlib import Path
import warnings
import logging

import torch

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    logging.disable()
    from gnn4itk_cf.core.core_utils import str_to_class
    logging.disable(logging.NOTSET)

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str)
parser.add_argument('-o', '--output', type=str, default="a.pt")
args = vars(parser.parse_args())

checkpoint_path = Path(args["checkpoint"])
assert checkpoint_path.exists()

config = torch.load(checkpoint_path)["hyper_parameters"]

module = str_to_class(config["stage"], config["model"]).load_from_checkpoint(checkpoint_path)
module.eval()

# Example input
n_nodes = 20
n_node_features = len(config["node_features"])
n_edges = 40

x = torch.ones(n_nodes,n_node_features)
edge_index = torch.randint(low=0, high=n_nodes, size=(2,n_edges))

# Trace
print("Tracing",config["stage"], config["model"])
if config["stage"] == "edge_classifier":
    script = module.to_torchscript(method='trace', example_inputs=(x, edge_index))
elif config["stage"] == "graph_construction":
    script = module.to_torchscript(method='trace', example_inputs=(x,))

torch.jit.save(script, args["output"])
