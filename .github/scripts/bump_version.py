"""Prepare a synchronized version bump; Cargo updates the lockfile afterward."""

import argparse
import datetime
import os
import re
from pathlib import Path

import tomllib


def prepare(root: Path, bump: str, requested: str = "") -> str:
    manifests = {
        "Cargo.toml": "package",
        "python/Cargo.toml": "package",
        "pyproject.toml": "project",
    }
    versions = {
        path: tomllib.loads((root / path).read_text())[section]["version"]
        for path, section in manifests.items()
    }
    current = versions["Cargo.toml"]
    if len(set(versions.values())) != 1:
        raise ValueError(f"Manifest versions disagree: {versions}")
    pattern = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    if not re.fullmatch(pattern, current):
        raise ValueError("Current version must be a stable MAJOR.MINOR.PATCH version")
    parts = tuple(map(int, current.split(".")))
    if requested:
        if not re.fullmatch(pattern, requested):
            raise ValueError("Requested version must be a stable MAJOR.MINOR.PATCH version")
        new = requested
    elif bump == "major":
        new = f"{parts[0] + 1}.0.0"
    elif bump == "minor":
        new = f"{parts[0]}.{parts[1] + 1}.0"
    elif bump == "patch":
        new = f"{parts[0]}.{parts[1]}.{parts[2] + 1}"
    else:
        raise ValueError("Unknown bump type")
    if tuple(map(int, new.split("."))) <= parts:
        raise ValueError("New version must be greater than the current version")

    changes = {}
    for path in manifests:
        source = (root / path).read_text()
        # Restrict replacement to the version matching the validated manifest.
        updated, count = re.subn(
            rf'^version = "{re.escape(current)}"$', f'version = "{new}"', source, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(f"Expected exactly one package version in {path}")
        changes[path] = updated
    path = "python/src/lib.rs"
    source = (root / path).read_text()
    updated, count = re.subn(
        rf'm\.add\("__version__", "{re.escape(current)}"\)',
        f'm.add("__version__", "{new}")',
        source,
    )
    if count != 1:
        raise ValueError("Python module version disagrees with manifests")
    changes[path] = updated
    changelog = (root / "changelog.md").read_text()
    if changelog.count("## Unreleased\n") != 1 or re.search(
        rf"^## {re.escape(new)}(?:\s|$)", changelog, re.MULTILINE
    ):
        raise ValueError("Expected one Unreleased section and no existing target release")
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    changes["changelog.md"] = changelog.replace(
        "## Unreleased\n", f"## Unreleased\n\n## {new} - {today}\n", 1
    )
    # Validate everything before writing any file.
    for path, content in changes.items():
        (root / path).write_text(content)
    return new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump", choices=["patch", "minor", "major"], default="patch")
    parser.add_argument("--version", default="")
    args = parser.parse_args()
    version = prepare(Path.cwd(), args.bump, args.version)
    print(version)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"version={version}\n")
