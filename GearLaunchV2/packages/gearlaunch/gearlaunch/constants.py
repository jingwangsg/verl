import yaml
import os

from .utils import run_cmd

POOLS = {
    # "h1": "groot-h100-01",
    # "h2": "groot-h100-02",
    "g1": "groot-gb200-01",
    "gc1": "groot-gb200-ci-01",
    "hl1": "groot-h100-large-01",
    "hl2": "groot-h100-large-02", 
    "hm1": "groot-h100-medium-01",
    "hm2": "groot-h100-medium-02",
    # "hs1": "groot-h100-small-01",
    "hc1": "groot-h100-ci-01",
    "hs2": "groot-h100-small-02",
    "l1": "groot-l40-01",
    "l2": "groot-l40-02",
    "l3": "groot-l40-03",
    "lc3": "groot-l40-ci-03",
    "l4": "groot-l40-04",
    "l1s": "groot-l40s-01",
}
DEFAULT_POOL = yaml.safe_load(run_cmd("osmo profile list").stdout)["pool"]["default"]
DEFAULT_POOL = [k for k, v in POOLS.items() if v == DEFAULT_POOL][0]


CACHE_ROOT = os.getenv("CACHE_ROOT", "/mnt/amlfs-02/shared/osmo/")

PROJECTS = ["trinity", "longrun", "neuralsim"]

RESOURCE_LIMIT = {
    "groot-gb200-01": {
        "gpu": 4,
        "storage": 18856,
        "cpu": 143,
        "memory": 873,
    },
    "groot-gb200-ci-01": {
        "gpu": 4,
        "storage": 18856,
        "cpu": 143,
        "memory": 873,
    },
    "groot-h100-large-01": {
        "gpu": 8,
        "storage": 892,
        "cpu": 93,
        "memory": 1864,
    },
    "groot-h100-ci-01": {
        "gpu": 8,
        "storage": 892,
        "cpu": 93,
        "memory": 1864,
    },
    "groot-h100-large-02": {
        "gpu": 8,
        "storage": 892,
        "cpu": 93,
        "memory": 1864,
    },
    "groot-h100-medium-01": {
        "gpu": 8,
        "storage": 892,
        "cpu": 93,
        "memory": 1864,
    },
    "groot-h100-medium-02": {
        "gpu": 8,
        "storage": 892,
        "cpu": 93,
        "memory": 1864,
    },
    # "groot-h100-small-01": {
    #     "gpu": 8,
    #     "storage": 892,
    #     "cpu": 93,
    #     "memory": 1864,
    # },
    "groot-h100-small-02": {
        "gpu": 8,
        "storage": 892,
        "cpu": 93,
        "memory": 1864,
    },
    "groot-l40-01": {
        "gpu": 8,
        "storage": 6333,
        "cpu": 127,
        "memory": 1006,
    },
    "groot-l40-02": {
        "gpu": 8,
        "storage": 6333,
        "cpu": 127,
        "memory": 1006,
    },
    "groot-l40-03": {
        "gpu": 8,
        "storage": 6332,
        "cpu": 126,
        "memory": 988,
    },
    "groot-l40-ci-03": {
        "gpu": 8,
        "storage": 6332,
        "cpu": 126,
        "memory": 988,
    },
    "groot-l40-04": {
        "gpu": 8,
        "storage": 6332,
        "cpu": 126,
        "memory": 988,
    },
    "groot-l40s-01": {
        "gpu": 4,
        "storage": 434,
        "cpu": 223,
        "memory": 1006,
    },
}
