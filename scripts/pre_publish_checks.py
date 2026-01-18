# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru==0.7.3",
#     "packaging==24.2",
#     "tomli==2.2.1 ; python_version < '3.11'",
# ]
# ///
import re
import subprocess
import sys
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path

from loguru import logger
from packaging.version import Version

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


logger.remove()
logger.add(sys.stderr, colorize=True, format="<level>{level:<5} {message}</level>")

REV_REGEXP = re.compile(
    r"(?P<prefix>https://raw\.githubusercontent\.com/neutrinoceros/rlic/)[^/]*(?P<suffix>.*)",
)
STABLE_VER_REGEXP = re.compile(r"^\d+\.*\d+\.\d+$")
STABLE_TAG_REGEXP = re.compile(r"^v\d+\.*\d+\.\d+$")
ROOT = Path(__file__).parents[1]
README = ROOT / "README.md"
PYPROJECT_TOML = ROOT / "pyproject.toml"
CARGO_TOML = ROOT / "Cargo.toml"


@dataclass(frozen=True)
class Metadata:
    current_python_static_version: Version
    current_rust_static_version: Version
    latest_git_tag: str

    @property
    def latest_git_version(self) -> Version:
        if not STABLE_TAG_REGEXP.match(self.latest_git_tag):
            logger.error(f"Failed to parse git tag (got {self.latest_git_tag})")
            raise SystemExit(1)
        return Version(self.latest_git_tag)


def check_static_version(md: Metadata) -> int:
    if not STABLE_VER_REGEXP.match(str(md.current_python_static_version)):
        logger.error(
            f"Current static version {md.current_python_static_version} doesn't "
            "conform to expected pattern for a stable sem-ver version.",
        )
        return 1
    elif md.current_python_static_version < md.latest_git_version:
        logger.error(
            f"Current static version {md.current_python_static_version} appears "
            f"to be older than latest git tag {md.latest_git_tag}",
        )
        return 1
    elif md.current_python_static_version != md.current_rust_static_version:
        logger.error(
            f"Python package version {md.current_python_static_version} and "
            f"rust crate version {md.current_rust_static_version} differ."
        )
        return 1
    else:
        logger.info("Check static version: ok", file=sys.stderr)
        return 0


def check_readme(md: Metadata) -> int:
    text = README.read_text()
    if md.current_python_static_version.is_devrelease:
        expected_tag = md.latest_git_tag
    else:
        expected_tag = f"v{md.current_python_static_version}"
    expected = REV_REGEXP.sub(
        lambda match: f"{match.group('prefix')}{expected_tag}{match.group('suffix')}",
        text,
    )
    if text != expected:
        diff = "\n".join(
            line.removesuffix("\n")
            for line in unified_diff(
                text.splitlines(),
                expected.splitlines(),
                fromfile=str(README),
            )
        )
        logger.error(diff)
        return 1
    else:
        logger.info("Check README.md: ok")
        return 0


def main() -> int:
    with open(PYPROJECT_TOML, "rb") as fh:
        package_table = tomllib.load(fh)
        current_python_static_version = Version(package_table["project"]["version"])
    with open(CARGO_TOML, "rb") as fh:
        crate_table = tomllib.load(fh)
        current_rust_static_version = Version(crate_table["package"]["version"])

    cp = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        check=True,
        capture_output=True,
    )
    cp_stdout = cp.stdout.decode().strip()

    md = Metadata(
        current_python_static_version,
        current_rust_static_version,
        cp_stdout,
    )

    return check_static_version(md) + check_readme(md)


if __name__ == "__main__":
    raise SystemExit(main())
