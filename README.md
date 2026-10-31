# 南京邮电大学离线校园问答系统

《大数据项目实践》课程项目。采集学校、职能部门和学院的公开资料，在本地使用 Qwen3-0.6B 完成校园问答，支持交互查询和批量 CSV 输出。

系统使用 BM25、BGE 向量检索、RRF 融合和 Reranker 查找证据。正文和结构化表格存入 SQLite；模型负责查询规划和回答组织，Python 校验来源、年份、单位并执行数值计算。资料不足时拒答，引用来自实际使用的资料。

## 安装

已验证环境：Windows、Python 3.11、NVIDIA RTX 4050 Laptop 6GB。CUDA 不是任务书的硬性要求；其他设备的速度和内存占用需实测。

```powershell
conda create -n campusrag python=3.11
conda activate campusrag
python -m pip install -e ".[dev,lora]"
```

已有合适的 Conda 环境时直接激活，无需重复创建，也不需要再使用 `venv`。以上安装命令在项目根目录执行。如使用 GPU，先安装与本机驱动匹配的 CUDA 版 PyTorch。默认启用数值规划 LoRA，需安装 `lora` 依赖并准备对应权重。

## 准备离线资源

GitHub 仓库包含项目代码，不包含大体积模型、数据库和索引。复现同一数据快照和答案，需要另行取得配套的 `campus-rag-resources.zip`，将其解压到项目根目录：

```text
models/Qwen3-0.6B/
models/bge-small-zh-v1.5/
models/bge-reranker-base/
models/adapters/planner_lora/step_300/
data/processed/campus_public/campus_public.db
data/processed/campus_public/clean_documents.jsonl
data/processed/campus_public/index/records.jsonl
data/processed/campus_public/index/vectors.npy
```

资源包的文件哈希和运行环境见 [资源清单](config/resources.json)。已有资源时无需重新采集、建索引或训练。请在新检出的项目目录解压，避免覆盖已有资料。

```powershell
Expand-Archive .\campus-rag-resources.zip -DestinationPath .
```

完整资源包解压后，可直接执行下面的资源检查和运行命令；不需要再下载模型、重新建库或训练。资源包 SHA256：

```text
983a1db86fe4cadf1e55f94a6804b83b8931ca3d31ecb7164f7d0f09eacded4d
```

可用 `Get-FileHash .\campus-rag-resources.zip -Algorithm SHA256` 核对。资源已存在的当前工作目录无需重复解压。

仅在没有配套资源、需要从头准备时，才联网下载基础模型：

```powershell
python scripts/prepare_models.py
```

数据采集、清洗、建库及从清洗快照恢复的步骤见 [数据准备](docs/data.md)。最终 LoRA 的训练方法见 [数值规划适配器](experiments/planner_lora/README.md)。实时重爬会受网页变化影响，不能保证还原同一快照。

## 运行

在项目根目录执行，资源检查通过后可以断网运行：

```powershell
python scripts/check_database.py
python scripts/check_offline.py
python app.py
```

批量回答：

```powershell
python scripts/batch_answer.py --input questions.csv --output results.csv
```

输入 CSV 使用 `问题` 列；输出为 UTF-8 with BOM，固定三列：

```text
问题,答案,参考资料
```

可用 `--audit-output audit.json` 另存计算和诊断。`app.py`、批量入口和离线检查均支持 `--config 配置路径`。交互和批量入口支持 `--no-adapter`；长期关闭 LoRA 可将配置的 `planner_adapter_path` 设为 `null`。关闭适配器可能影响数值规划效果。

## 验证

```powershell
python -m pytest -q
python scripts/batch_answer.py --input data/eval/campus_public/questions.csv --output results.csv
```

固定验收题覆盖任务书四例、校园事实、时效、拒答及学分学时计算。题集及复测方法见 [验收数据](data/eval/README.md)。这些是开发回归题，不能代表未知问题的正确率。

配套参考输出为 54 题中 53 题通过核对，任务书四例通过；贷款上限差额题被校验拦截并拒答。输出及核对摘要与当前正式源码、配置和数据资源的哈希绑定，详见验收数据说明。

对比基础模型与最终适配器，使用同一个正式批量入口：

```powershell
python experiments/planner_lora/verify_production.py --output .tmp/verification
```

该命令阻断推理期间的网络连接，输出两份三列 CSV、逐题核对及汇总；输出目录必须不存在。改变的文字答案需要人工复核。复测结果见该目录的 `comparison_summary.json`。

检索对照支持 BM25、纯向量、融合和重排四种模式：

```powershell
python scripts/evaluate_retrieval.py --dataset retrieval.jsonl
```

`retrieval.jsonl` 每行包含 `question` 和当前索引中的 `relevant_ids`。按文档内容键标注的来源题集及数据库间检索对比入口见 [验收数据](data/eval/README.md)。

## 文件用途

| 目录 | 内容 |
| --- | --- |
| `src/campus_rag/` | 采集、解析、数据库、检索、问答与计算 |
| `config/` | 运行参数、站点和固定数据构建配置 |
| `scripts/` | 资源准备、采集清洗、建库、检查、评测和批量入口 |
| `experiments/planner_lora/` | 最终适配器的数据生成、训练和验证 |
| `data/eval/` | 最终验收题及来源标签 |
| `tests/` | 当前功能的单元和集成回归测试，无需加载真实模型 |
| `docs/data.md` | 数据恢复与构建说明 |

当前快照包含 518 篇公开资料。支持 HTML、PDF、DOCX、XLSX、TXT、Markdown；扫描 PDF 不自动 OCR。多级表头和来源时效需要人工抽查，系统不保证回答任意校园问题。只采集公开页面，遵守站点白名单、访问间隔和 `robots.txt`。
