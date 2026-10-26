# 数据清洗与可靠入库

原始资料保存在 data/raw，不删除。清洗流程单独输出规范化正文、结构化表格、来源别名和每份资料的处理理由。文件基本可读不代表语义解析已经准确，招生计划、收费、师资等关键表格仍需人工抽查。

## 1 全量清洗

```powershell
conda activate rag
cd E:\CampusRAG
python scripts/clean_data.py
```

清洗先读取每个 URL 最新的采集状态，只处理成功资源。优先选择有发布日期的桌面版页面作为主来源；只合并清洗后内容完全相同、标题年份相同的文档，不做模糊合并。所有来源 URL 和元数据作为 sources 保留。缺少发布日期不自动排除，绝不以抓取日期冒充发布日期。

HTML 仅提取可靠正文容器内的内容，去除外围导航；表格单独解析，保留原始文字及行列位置。清洗空白和不可见字符，不删除数字、单位、日期及脚注。PDF、Word、Excel 使用原有解析器。

默认输出 data/processed/clean：

- clean_documents.jsonl：接收且去重后的正文、表格及完整来源。
- clean_decisions.jsonl：每个 URL 的 accept/duplicate/exclude/review 及理由。
- clean_report.json：计数与原因汇总。

review 表示待复核，不进入默认入库清单，包括旧 DOC/XLS、扫描或解析警告、正文容器无法识别、过短正文，以及非核心部门域名。review 不等于无用；后续应改进对应解析器或转换格式，再生成清洗清单。不提供自动猜测正文或强行绕过警告的开关。

正文仅承载 PDF 播放器的页面不会作为空文章入库，已下载的附件独立处理。广泛站点采集会包含非核心业务资料，extended_site_scope_check 应逐步按主题复核后扩充受信任站点。

## 2 小规模验证

```powershell
python scripts/clean_data.py --limit 40 --output data/processed/pilot_clean
python scripts/ingest_data.py --clean-manifest data/processed/pilot_clean/clean_documents.jsonl --database data/processed/pilot.db
python scripts/check_database.py --database data/processed/pilot.db
```

--limit 是部分试运行，不能把该报告误认为全量结果。pilot.db 与正式 campus.db 分离。

## 3 正式入库

```powershell
python scripts/ingest_data.py --clean-manifest data/processed/clean/clean_documents.jsonl
python scripts/check_database.py
```

每篇文档、表格、单元格、分块和导入完成标记在同一 SQLite 事务写入；异常则整篇回滚。已完成的相同内容重复导入不增加表格或分块，新增别名 URL 可以补入 document_sources。不同年份正文相同也不自动合并。全量清洗重新生成的文件，不会自动删除数据库里的旧记录或撤回资料，重新选集需使用独立数据库核对后切换。

如数据库来自旧版无 ingestion_state 标记的导入，程序会阻止重复追加相同文档，需备份旧库后在新数据库重建，不自动删除旧表。当前只支持源码可编辑安装，新增 schema 文件通过原有 initialize 自动创建。

分块使用本地 BGE tokenizer；正文和表格描述都执行硬上限。所有表格行参与检索描述分块，不再只保留前八行；真实计算仍读取 SQLite 完整表格。自动解析的多级表头、合并单元格、数值及单位仍需人工验证。

## 4 建立索引与人工验收

```powershell
python scripts/build_index.py
python scripts/check_offline.py
python app.py
```

入库内容改变后应重新构建索引。数据库结构检查通过不表示答案质量通过。至少抽查招生简章、收费章程、质量报告、学生手册和图书馆服务：对应正文是否去掉导航，表格首尾行和合计是否完整，年份和单位是否正确，引用是否指回原官方资料。部分 PDF 页可能含图像，即便其他页能提取文本也要检查是否漏了关键表格。
