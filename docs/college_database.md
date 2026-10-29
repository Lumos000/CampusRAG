# 518 篇 v3 数据集构建与验收

当前数据集由原库 497 篇文档和计算机学院 15 篇、人工智能学院 6 篇补采文档组成。构建仅创建新数据库和索引，默认应用配置仍保留原库。

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `config/sites_colleges.json` | 只跟踪计算机学院与人工智能学院官网的定向补采 |
| `config/sites_topics.json` | 五类主题与学院站点的通用探索配置 |
| `config/dataset_v3.json` | v3 固定输入、排除名单、518 篇数量及必需学院来源检查 |
| `scripts/collect_topics.py` | 独立采集、只读复用原始资料、清洗与覆盖报告 |
| `scripts/rebuild_college_database.py` | 合并去重、保护原库、排除指定资料、独立重建与索引 |
| `scripts/compare_college_retrieval.py` | 固定题集和模型的来源检索对比 |
| `scripts/smoke_qa.py` | 在指定数据库配置下运行实际问答样例 |
| `data/eval/college_dataset_v3/` | 可版本管理的固定题集、问答要点与历史快照摘要 |

## 输入与版本边界

原库为 `data/processed/campus.db`，对应清洗清单为 `data/processed/clean/clean_documents.jsonl`；补采清单为 `data/collections/colleges_20261029_v1/clean/clean_documents.jsonl`。
原始资料、清洗正文、数据库、模型及向量索引是本地生成物，不提交到 GitHub。新检出仓库需要先取得相应资料和本地模型；只有代码和配置不能恢复这份历史数据快照。

518 是已验收快照的数量，不承诺实时重爬仍得到相同数量。重建配置会检查预期数量及计算机、人工智能学院来源；数据变化时会在创建输出目录前停止，应先审核新数据，再建立新的版本配置。

v3 排除通信学院《学院简介》以及自动化学院《学院简介》《本科专业介绍》。这三篇的精确 URL 固定在构建配置中，避免遗漏命令行参数后重新加入。该流程不允许排除原库文档。

## 构建与启动

在项目根目录运行：

```powershell
conda activate rag
python scripts/rebuild_college_database.py --recipe config/dataset_v3.json --build-index
python scripts/check_database.py --database data/processed/colleges_20261029_v3/campus_colleges.db
python app.py --config data/processed/colleges_20261029_v3/settings.json
```

目标目录必须不存在。已有 v3 时可用 `--output data/processed/colleges_v3_review` 指定全新的验证目录；不要删除原库来重跑。输入路径按项目根目录解析，命令行路径可覆盖配置。

重建先核对原库内容键与原清洗清单一致，再合并补采资料。同一原始 URL、哈希出现不同解析结果时，保留原库表示并合并来源别名。每篇文档沿用原有事务导入，表格规范化只作用于新库。

输出包含 `campus_colleges.db`、`clean_documents.jsonl`、`settings.json`、`rebuild_report.json`，以及启用 `--build-index` 时的 `index/`。报告记录输入清单哈希、原库前后哈希、排除项、来源分布和结构检查；导入或索引失败的目录不能作为已完成版本使用。

学院补采仍使用独立新目录：

```powershell
python scripts/collect_topics.py --config config/sites_colleges.json --max-pages 80 --max-depth 2 --output data/collections/colleges_next
```

网页变化会影响新增资料数量，应先检查清洗决定与覆盖报告。更详细的采集规则见 [独立补采说明](topic_collection.md)。

## 复测与历史结果

版本化题集包含 14 道原有正例留出题、10 道原主题题、17 道学院题，以及 8 道实际问答样例。
检索按文档内容键标注来源，避免依赖每次重建变化的分块 UUID；加载时核对全部分块 ID、正文及来源元数据，防止索引和数据库混用。

```powershell
python scripts/compare_college_retrieval.py --candidate-config data/processed/colleges_20261029_v3/settings.json --cases data/eval/college_dataset_v3/cases.json --output data/collections/v3_retrieval_review
python scripts/smoke_qa.py --config data/processed/colleges_20261029_v3/settings.json --questions-file data/eval/college_dataset_v3/smoke_questions.json --output data/collections/v3_smoke_review.json
python -m pytest tests/test_college_rebuild.py tests/test_crawler.py tests/test_clean_ingest.py tests/test_table_parser.py tests/test_database_calculation.py tests/test_numeric.py -q
```

历史本地快照摘要见 `data/eval/college_dataset_v3/verified_snapshot.json`；它不是本次提交的端到端正确率保证。原始数据和模型版本发生变化后必须重新评测。

| 历史检查项 | v3 结果 |
| --- | --- |
| 文档 / 分块 / 表格 / 来源链接 | 518 / 13968 / 811 / 852 |
| 来源 | 学校、五个职能部门及两个必需学院，共 8 个官方站点 |
| 原库保留 | 原文档、分块内容、来源别名、原始表格单元格全部保留 |
| 原始文件 | 517 个不同文件的哈希全部匹配 |
| 数据库 | 完整性、外键、导入完成标记检查通过，最大分块 479 tokens |
| 原有 24 题重排来源 Hit@5 | 21/24 |
| 学院 17 题重排来源 Hit@5 / Hit@10 | 16/17 / 17/17 |
| 实际问答要点检查 | 4/8，学校前身回答仍有无关内容 |

数据结构、官方来源和五类主题基本覆盖适合后续校园 RAG 开发实验；当前问答效果仍需改进。已观察到本科专业与微专业混淆、学分问题选错表格、选课问题误入数值路径和续借条款遗漏，相关原文在库中均可查到。

来源命中率不是答案正确率。题集样本量有限，同一助手编写和标注；136 篇文档缺少发布日期，历史专业介绍、办事表格和临时通知须核对适用时间。本次整理不修改规划器、检索算法或回答逻辑。
