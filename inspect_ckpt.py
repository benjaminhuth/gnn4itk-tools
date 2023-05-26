#!/bin/python3

import argparse
from pathlib import Path
import datetime

import torch


parser = argparse.ArgumentParser()
parser.add_argument("ckpt", type=str)
args = vars(parser.parse_args())

ckpt_file = Path(args["ckpt"])
assert ckpt_file.exists()

ckpt = torch.load(ckpt_file)

hparams = ckpt["hyper_parameters"]

print("modified", datetime.datetime.fromtimestamp(ckpt_file.lstat().st_mtime))
print("state", hparams["stage"])
print("model", hparams["model"])
