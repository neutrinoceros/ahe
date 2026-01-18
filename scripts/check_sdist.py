import sys
from pathlib import Path


def main() -> int:
    dist_dir = Path(__file__).parents[1] / "dist"
    sdist_files = sorted(dist_dir.glob("*.tar.gz"))
    if not sdist_files:
        print("No source distribution found", file=sys.stderr)
        return 1
    if len(sdist_files) != 1:
        print("Found more than one source distribution", file=sys.stderr)
        return 1

    sdist = sdist_files[0]
    if (size := sdist.stat().st_size) > (max_size := 50_000):
        print(
            f"Source distribution size ({size:_}) exceeds expected limit ({max_size:_})",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
