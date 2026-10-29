from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_rag.config import load_settings
from campus_rag.index import HybridIndex
from campus_rag.rag import RAGService
from campus_rag.rag.generator import TransformersGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description='南邮离线校园问答')
    parser.add_argument('--config', type=Path, help='指定数据库和索引配置，默认使用原配置')
    args = parser.parse_args()
    settings = load_settings(args.config)
    index = HybridIndex.load(settings.index_dir, settings.embedding_model, settings.reranker_model)
    generator = TransformersGenerator(settings.generator_model)
    connection = sqlite3.connect(settings.database_path)
    try:
        service = RAGService(settings, index, generator, connection)
        print("南邮离线校园问答。输入 exit 退出。")
        while True:
            question = input("\n问题> ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                break
            result = service.answer(question)
            print(f"\n{result.answer}")
            if result.references:
                print("\n参考资料：")
                for reference in result.references:
                    print(f"- {reference}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
