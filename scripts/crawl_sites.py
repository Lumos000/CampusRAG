from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from campus_rag.config import ROOT
from campus_rag.crawl import CampusCrawler


def main() -> None:
    parser = argparse.ArgumentParser(description="采集白名单中的公开校园资料")
    parser.add_argument("--config", default=ROOT / "config" / "sites.json")
    parser.add_argument("--output", default=ROOT / "data" / "raw")
    parser.add_argument("--max-pages", type=int, help="本次最多请求的资源数")
    parser.add_argument("--max-depth", type=int, help="从种子页面开始的最大跟踪深度")
    parser.add_argument('--refresh', action='store_true', help='忽略续爬队列，从种子重新检查更新')
    args = parser.parse_args()
    if args.max_pages is not None and args.max_pages < 1:
        parser.error('--max-pages 必须大于 0')
    if args.max_depth is not None and args.max_depth < 0:
        parser.error('--max-depth 不能小于 0')
    count = CampusCrawler(args.config, args.output, args.max_pages, args.max_depth, args.refresh).run()
    print(f"已保存 {count} 个公开资源。")
    print(f"采集清单和统计报告位于：{args.output}")


if __name__ == "__main__":
    main()
