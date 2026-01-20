# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.1",
# ]
# ///

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def stylesheet(theme: Literal["light", "dark"]) -> Path:
    return Path(__file__).parents[1] / "assets" / f"{theme}.mplstyle"


# assuming
# cp311-cp311-macosx_11_0-arm64

# https://pypi.org/project/numpy/#files
# 2.4.1
NUMPY_WHL_SIZE = 13_372

skimage_env_sizes = {
    # https://pypi.org/project/scikit-image/#files
    # 0.26.0
    "scikit-image": 12_304,
    # 12.1.0
    "pillow": 4_552,
    # 2.37.2
    "imageio": 312,
    # 0.4
    "lazy-loader": 12,
    # 25.0
    "packaging": 68,
    # 3.6.1
    "networkx": 2_024,
    # 1.17.0
    "scipy": 31_456,
    # 2026.1.14
    "tifffile": 228,
}

wheels_sizes = {
    "scikit-image (with deps*)": sum(skimage_env_sizes.values()),
    "scikit-image (by itself)": skimage_env_sizes["scikit-image"],
    # 0.0.1
    "ahe": 216,
}

sizes = np.array(list(wheels_sizes.values()), dtype="int64")

data = {
    "name": list(wheels_sizes),
    "size": sizes,
    "normalized_size": sizes.astype("float64") / NUMPY_WHL_SIZE,
}

sizes_MiB = sizes.astype("float64") / 1024


def plot(theme: Literal["light", "dark"]) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.8))
    ax.barh("name", "normalized_size", data=data, height=0.5, color="C4")
    # ax.invert_yaxis()
    ax.set(title="wheel size (normalized by numpy's)")
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    sfile = Path(__file__).parents[1] / "assets" / f"wheel-size-{theme}.svg"
    print(f"saving to {sfile}")
    fig.savefig(sfile, bbox_inches="tight", transparent=True)


def main() -> int:
    for theme in ["light", "dark"]:
        with plt.style.context(stylesheet(theme)):
            plot(theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
