#!/usr/bin/env python3
"""Reuse native extensions from an ABI-identical editable vLLM build.

The research branch changes Python only.  A fresh clone may therefore have no
``*.so`` files even when the selected environment already contains an editable
build of the same native sources.  This helper verifies every tracked native
source byte before linking those build artifacts into the current tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse


TRACKED_NATIVE_ROOTS = ("CMakeLists.txt", "cmake", "csrc")


def editable_source() -> Path:
    dist = importlib.metadata.distribution("vllm")
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        raise SystemExit("installed vLLM has no editable source metadata")
    metadata = json.loads(direct_url)
    if not metadata.get("dir_info", {}).get("editable"):
        raise SystemExit("installed vLLM is not an editable build")
    parsed = urlparse(metadata["url"])
    if parsed.scheme != "file":
        raise SystemExit(f"unsupported editable URL: {metadata['url']}")
    return Path(unquote(parsed.path)).resolve()


def tracked_native_digest(repo: Path) -> tuple[str, list[str]]:
    command = ["git", "-C", str(repo), "ls-files", "--", *TRACKED_NATIVE_ROOTS]
    files = subprocess.run(command, check=True, text=True, capture_output=True).stdout.splitlines()
    digest = hashlib.sha256()
    for relative in files:
        path = repo / relative
        if not path.is_file():
            raise SystemExit(f"tracked native source is missing: {path}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", default=os.environ.get("Q38NEXT_NATIVE_ARTIFACT_SOURCE"))
    args = parser.parse_args()

    target = Path(args.repo).resolve()
    source = Path(args.source).resolve() if args.source else editable_source()
    if source == target:
        return

    target_digest, target_files = tracked_native_digest(target)
    source_digest, source_files = tracked_native_digest(source)
    if target_files != source_files or target_digest != source_digest:
        raise SystemExit(
            "refusing native artifact reuse: tracked CMake/cmake/csrc sources differ\n"
            f"  target={target} digest={target_digest}\n"
            f"  source={source} digest={source_digest}"
        )

    artifacts = sorted((source / "vllm").glob("*.so"))
    artifacts += sorted((source / "vllm" / "vllm_flash_attn").glob("*.so"))
    generated_dirs = [
        source / "vllm" / "vllm_flash_attn" / name
        for name in ("cute", "layers", "ops")
    ]
    generated_dirs += [
        source / "vllm" / "third_party" / name
        for name in ("triton_kernels", "flashmla", "deep_gemm", "fmha_sm100", "tml_fa4")
    ]
    if not artifacts or not all(path.is_dir() for path in generated_dirs):
        raise SystemExit(f"editable source has no complete native build artifacts: {source}")

    linked = []
    for artifact in artifacts:
        relative = artifact.relative_to(source)
        destination = target / relative
        if destination.exists() or destination.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(artifact)
        linked.append(str(relative))
    for directory in generated_dirs:
        relative = directory.relative_to(source)
        destination = target / relative
        if not destination.exists() and not destination.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(directory)
            linked.append(str(relative))
            continue
        # A fresh source tree can retain a tracked __init__.py for a generated
        # package. Link only missing generated children in that case.
        if not destination.is_dir():
            raise SystemExit(f"generated-package destination is not a directory: {destination}")
        for child in directory.iterdir():
            if child.name == "__pycache__":
                continue
            child_destination = destination / child.name
            if child_destination.exists() or child_destination.is_symlink():
                continue
            child_destination.symlink_to(child)
            linked.append(str(child_destination.relative_to(target)))

    print(json.dumps({
        "status": "passed",
        "target": str(target),
        "source": str(source),
        "native_source_sha256": target_digest,
        "linked": linked,
    }, indent=2))


if __name__ == "__main__":
    main()
