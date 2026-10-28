# 南京邮电大学离线校园问答系统

这是《大数据项目实践》的离线 RAG 实现。系统将公开校园资料解析为正文块和结构化 SQLite 表格，通过 BM25、BGE 向量检索、RRF 和小型 Reranker 找到证据。Qwen3-0.6B 只负责查询规划和回答组织；Python 负责参数化筛选、单位验证、Decimal 计算以及参考资料生成。

版本与评测入口见 [代码与版本核对记录（2026-10-27）](docs/current_code_status.md)，包括本地与 GitHub 提交差异、模块职责及当前可用边界。

上一轮可靠性优化见 [修改与验证报告](docs/reliability_optimization_v1.md)及 [36 题逐题对照](docs/reliability_case_comparison.md)。冻结开发诊断的核心要点评分由 24/36 增至 33/36；文字回答主要依赖保守原文回退，平均长度明显增加，数值规划仍为 5/8；当时独立 v2 尚未评测。

候选优化见 [小范围优化与淘汰记录](docs/reliability_trials_v2.md)：保持开发诊断33/36及数值5/8，文字/时效题平均长度从610字降至313字，141项自动测试与12题回归通过。无收益的数值约束生成已撤回，该版本冻结为 `10b9ceb`。

该提交的 [独立v2首次评测](docs/independent_v2_evaluation_10b9ceb.md) 严格正确6/16，数值0/8，普通文字3/4，时效2/2，暴露了Facts抽取、对象绑定、误路由及漏算人数的语义校验问题。首次结果与原金标保留，不能把开发成绩作为通用问答可靠率。

后续 [检索后处理修复与逐题回归](docs/post_retrieval_reliability_fix_v1.md)：165项自动测试、12题回归及任务书四例通过；36题保持33/36；已公开v2回归为13/16，数值5/8。v2已用于调试，这不是新的独立准确率；剩余问题包括规划漏操作数及文字回答冗长。数据库和索引无需重建。

## 运行环境

- Windows 10/11，Python 3.10 或 3.11。
- NVIDIA RTX 4050 Laptop GPU 6GB，建议使用支持当前显卡驱动的 CUDA 版 PyTorch。
- 正式运行不需要网络；下载依赖、模型和采集公开资料属于准备阶段。

创建环境并安装项目：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 如需 CUDA，请先按 PyTorch 官网给出的当前命令安装对应版本 torch。
python -m pip install -e ".[dev]"
```

不要把密码、Cookie、Token 或登录后的网页数据放入项目。本系统只采集公开页面，并遵守域名白名单、访问间隔和 `robots.txt`。

## 准备模型

联网准备阶段运行：

```powershell
python scripts/prepare_models.py
```

模型保存为：

```text
models/Qwen3-0.6B/
models/bge-small-zh-v1.5/
models/bge-reranker-base/
```

正式运行通过 `local_files_only=True` 加载模型，并设置 Hugging Face/Transformers 离线环境变量。Qwen3-0.6B 的 BF16 权重应能放入 6GB 显存；若实际机器上显存不足，可另行准备 Q5_K_M 或 Q6_K GGUF，并使用本地 `llama-server` 适配生成器。默认实现采用 Transformers，避免第一版同时维护两套推理后端。

## 数据准备

采集完成后按 [数据清洗与可靠入库](docs/cleaning.md) 执行 clean_data.py、清洗清单导入和数据库校验；不建议把原始网页清单直接全量入库。

爬虫参数、采集清单字段和试运行步骤见 [采集操作说明](docs/crawling.md)。

先检查并扩充 [config/sites.json](config/sites.json) 中的公开站点种子，然后运行：

```powershell
python scripts/crawl_sites.py
python scripts/clean_data.py
python scripts/ingest_data.py --clean-manifest data/processed/clean/clean_documents.jsonl
python scripts/check_database.py
python scripts/build_index.py
```

也可以直接导入人工下载的文件：

```powershell
python scripts/ingest_data.py "data/raw/example.pdf" "data/raw/example.xlsx"
```

支持 HTML、PDF、DOCX、XLSX、TXT 和 Markdown。正文使用本地 BGE tokenizer 分块，目标 280～420 tokens，硬上限 480 tokens。HTML/XLSX 合并单元格会展开；重复的上层表头与下一行组合为多级表头。PDF 扫描页或表格解析失败会写入 `audit_issues`，需要人工检查。

SQLite 位于 `data/processed/campus.db`，其中原始单元格值、规范化值、数值、单位及来源位置分别保存。表格描述块只参与召回，最终筛选和计算从结构化表中读取。

## 离线问答

完成模型、数据库和索引准备后，先断开网络并执行：

```powershell
python scripts/check_offline.py
python app.py
```

未知测试集批处理：

```powershell
python scripts/batch_answer.py --input data/eval/questions.csv --output results.csv
```

输入 CSV 至少包含一列问题，推荐列名为 `问题`。输出固定为 UTF-8 with BOM 编码，列顺序严格是：

```text
问题,答案,参考资料
```

参考资料由实际参与回答或计算的数据库记录自动生成，Qwen 的输出不会被当作引用。

## 查询和计算安全边界

新的通用数值路径见 [受限分步数值推理](docs/step_reasoning.md)：先检索，再向 Qwen 提供真实 Schema，使用字段 ID 和 JSON 步骤在 SQLite 全表执行，直接输出程序核验的结果与来源。既有数据库请先运行 `python scripts/normalize_tables.py`；该操作自动备份并保留原始数据及所有 ID，不需要重建当前索引。

当前新增的数值核验支持专业类别学费及上浮比例、住宿费与多年合计、师资人数及折算/生师比、首选/再选科目筛选招生计划。来源缺失、同一字段冲突、选科关系不明确时拒绝计算。核验日志保留原文数值和单元格位置；最终数值回答使用经过校验的模板，不接受模型任意改写数值或字段。详见 [数值核验说明](docs/numeric_validation.md)。

数值计划必须通过 JSON、字段、范围、年份、单位和完整性校验。格式或计划错误最多共享一次修复机会；证据或统计条件不足时明确拒答。普通文字事实题不预先调用规划器。模型不能提交 SQL 或 Python，已核验的数值答案不再交给模型改写。

结构化查询仅支持白名单筛选操作 `contains/eq/gt/gte/lt/lte`，以及 `sum/count/avg/ratio/difference/multiply/average`。Python 将字段映射为数据库列并使用参数化查询；数值使用 `Decimal`。多值歧义、空结果、单位冲突、分母为零和不支持的单位转换都会拒绝计算。

## 评测

运行核心测试：

```powershell
python -m pytest -q
```

检索评测集采用 JSONL，每行包括 `question` 和 `relevant_ids`：

```json
{"question":"图书馆开放时间是什么？","relevant_ids":["实际chunk_id"]}
```

执行：

```powershell
python scripts/evaluate_retrieval.py --dataset data/eval/retrieval.jsonl
```

至少比较 BM25、纯向量、RRF 融合、融合加小型 Reranker 四组配置，并记录 Recall@5/10/20、MRR、答案准确率、数值准确率、引用支持率、拒答准确率、耗时和峰值显存。

## 目录

```text
config/                 站点和运行参数
data/raw/               原始公开资料及 manifest
data/processed/         SQLite 数据库和处理产物
data/index/             向量与检索记录
data/eval/              测试集
models/                 完整离线模型
src/campus_rag/crawl/   白名单爬虫
src/campus_rag/ingest/  文档解析、token 分块和表格导入
src/campus_rag/index/   BM25、向量、RRF、Reranker
src/campus_rag/database/SQLite schema 和受限查询
src/campus_rag/rag/     查询规划、回答与系统引用
src/campus_rag/tools/   数字、单位和计算工具
scripts/                准备、建库、评测和批处理入口
tests/                  单元与集成测试
```

## 当前边界

- 扫描版 PDF 不自动 OCR，系统会报告并要求人工补录；这比静默产生错误文本更可靠。
- 多级表头采用通用启发式，重要招生和质量报告应在导入后抽查 SQLite。
- 默认只启用小型 Reranker。只有消融实验证明更大的模型有稳定收益，才应把它作为实验分支。
- 当前不包含 QLoRA。应先完成可追溯数据、检索和计算基线，避免用微调掩盖检索错误。

