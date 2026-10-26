from __future__ import annotations

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401


MODELS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
    "bge-reranker-base": "BAAI/bge-reranker-base",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="联网准备阶段：下载正式运行所需模型")
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    # Xet large-file transfers can stall on some Windows/campus networks.
    # Standard HTTP is slower but easier to resume and diagnose.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    from huggingface_hub import snapshot_download

    args.output.mkdir(parents=True, exist_ok=True)
    for directory, repo in MODELS.items():
        target = args.output / directory
        print(f"下载 {repo} -> {target}")
        snapshot_download(
            repo_id=repo,
            local_dir=target,
            max_workers=1,
            ignore_patterns=["*.bin", "*.onnx", "*.h5", "*.ot"],
        )
    print("模型准备完成。正式运行前请执行 scripts/check_offline.py。")


if __name__ == "__main__":
    main()
