# 数据准备

默认配置使用 `data/processed/campus_public/campus_public.db` 和同目录的 `index/`。恢复资源包后先执行 `scripts/check_database.py` 和 `scripts/check_offline.py`。

## 从清洗快照恢复

配套资源的 `clean_documents.jsonl` 保存解析后的正文、表格和公开来源，可以不访问原网站重新入库。直接使用资源包的数据库及匹配索引，可保留验收题绑定的文档和字段 ID。

如需验证清洗快照能否重建，在独立目录执行：

```powershell
python scripts/ingest_data.py --clean-manifest data/processed/campus_public/clean_documents.jsonl --database data/processed/rebuild/campus_public.db
python scripts/check_database.py --database data/processed/rebuild/campus_public.db
```

重新导入可能改变分块 UUID 和数据库 ID，不能搭配旧索引。复制 `config/settings.json` 为独立配置，设置新数据库和索引路径，再执行 `python scripts/build_index.py --config 配置路径`。按新配置运行应用，并重新核对来源标签。

## 采集新资料

```powershell
python scripts/crawl_sites.py
python scripts/clean_data.py
```

站点和采集限制见 `config/sites.json`。检查 `data/processed/clean/` 的清洗决定和待复核记录后再导入：

```powershell
python scripts/ingest_data.py --clean-manifest data/processed/clean/clean_documents.jsonl
python scripts/check_database.py
python scripts/build_index.py
```

正文按本地 BGE tokenizer 分块，硬上限 480 tokens。清洗去除错误页及重复内容；表格保留原始单元格和规范化值，扫描 PDF 及解析问题需人工处理。

定向补采学院或主题资料，可用独立入口，输出目录必须不存在：

```powershell
python scripts/collect_topics.py --config config/sites_colleges.json --max-pages 80 --max-depth 2 --output data/collections/colleges_next
```

`config/sites_topics.json` 提供通用主题配置。独立补采只采集和清洗，不改动正式数据库。

## 固定数据构建

`config/dataset_campus_public.json` 从基线及学院补采清单构建当前 518 篇资料库。需同时提供其中指定的基线数据库、基线清洗清单和补采清洗清单；配置文件本身不包含数据。

```powershell
python scripts/rebuild_college_database.py --recipe config/dataset_campus_public.json --build-index
```

构建核对基线内容、合并去重、检查排除 URL、文档数量和必需学院来源。输出目录必须不存在；已有默认库时，用 `--output data/processed/campus_public_review` 选择新目录，运行时使用生成的 `settings.json`。

旧数据库可用 `python scripts/normalize_tables.py --database 数据库路径` 升级派生表格元数据；自动备份，保留原始内容及 ID。正式问答始终只读数据库。更新资料或重建索引后必须重新评测。
