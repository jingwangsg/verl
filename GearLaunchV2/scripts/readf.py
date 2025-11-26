import torch
import json
import yaml
# import pandas as pd
import polars as pl
import argparse
import ipdb
import h5py
import numpy as np
import dill

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the pickle file")
    args = parser.parse_args()

    path = args.path

    if path.endswith(".pkl") or path.endswith(".pickle") or path.endswith(".pt"):
        try:
            data = torch.load(args.path, map_location="cpu", weights_only=False)
        
        except Exception as e:
            print(f"Error loading {args.path}: {e}, trying dill...")
            data = dill.load(open(args.path, "rb"))
    elif path.endswith(".json"):
        with open(args.path, "r") as f:
            data = json.load(f)
    elif path.endswith(".yaml") or path.endswith(".yml"):
        with open(args.path, "r") as f:
            data = yaml.load(f)
    elif path.endswith(".csv"):
        data = pl.read_csv(args.path)
    elif path.endswith(".parquet"):
        data = pl.read_parquet(args.path)
    elif path.endswith(".h5") or path.endswith(".hdf5"):
        data = h5py.File(args.path, "r")
    elif path.endswith(".npz") or path.endswith(".npy"):
        data = np.load(args.path)
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    print(f"class(data): {data.__class__}")

    ipdb.set_trace()
