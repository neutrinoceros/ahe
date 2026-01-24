import json
from functools import partial
from pathlib import Path
from time import monotonic_ns

import numpy as np
from skimage.exposure import equalize_adapthist, equalize_hist

import ahe

BENCHMARK_DIR = Path(__file__).parent

SHAPE = 2028, 2048
prng = np.random.default_rng(0)
image = np.clip(
    prng.normal(loc=0.0, scale=0.25, size=np.prod(SHAPE)), a_min=-1.0, a_max=1.0
).reshape(SHAPE)  # np.load(BENCHMARK_DIR / "img.npy")


def loop(func, n: int):
    for _ in range(n):
        func()


impls = {
    "simple": {
        "scikit-image": partial(equalize_hist, image),
        "ahe": partial(ahe.equalize_histogram, image, nbins=256),
    },
    "adaptive": {
        "scikit-image": partial(equalize_adapthist, image, clip_limit=0.0),
        "ahe": partial(
            ahe.equalize_histogram,
            image,
            nbins=256,
            adaptive_strategy={"kind": "tile-interpolation", "tile-into": 8},
        ),
    },
}


def main() -> int:
    results = {}
    for kind, funcs in impls.items():
        results.setdefault(kind, {})
        for pkg, f in funcs.items():
            tstart = monotonic_ns()
            loop(f, n=100)
            tstop = monotonic_ns()
            results[kind][pkg] = tstop - tstart
    sfile = BENCHMARK_DIR / "results.json"
    print(f"saving results to {sfile}")
    with open(sfile, "w") as fp:
        json.dump(results, fp, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
