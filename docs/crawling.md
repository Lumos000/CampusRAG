# 公开校园资料采集

在项目根目录的 rag 环境运行。初次试运行：

```powershell
python scripts/crawl_sites.py --max-pages 20 --max-depth 1
```

正式扩大采集：

```powershell
python scripts/crawl_sites.py --max-pages 200 --max-depth 3
```

max-pages 是本次最多尝试的 URL 数，包括失败和 robots 拒绝项，不承诺得到同样数量的有效资料。程序在 crawl_state.json 保存已访问 URL 和待采集队列，同样配置和深度下再次运行会继续采集新链接，失败项留待下一轮重试。配置或深度改变时自动从种子重新探索。使用 --refresh 可重新检查已抓页面，通过 ETag/Last-Modified 和内容哈希减少重复存储。运行时逐个成功资源保存检查点，异常中断最多重做最近的未保存工作。

建议分批扩大，例如重复执行以下命令，每次最多尝试 500 个新 URL：

```powershell
python scripts/crawl_sites.py --max-pages 500 --max-depth 4
python scripts/audit_raw.py
```

质量审计保存在 audit_report.json，检查文件哈希、PDF/Office 可读性、错误提示页和缺失日期。usable_snapshots 仅代表文件基本有效，不代表与题目相关或已完成语义解析。首页/栏目页日期为空是正常现象，政策文章日期为空需人工核实。当前新增了官方图书馆服务流程、借阅规则、读者服务、新读者向导，以及招生计划、综合评价、学籍管理、学生资助入口；仍需逐步补充学院主题。

配置文件 config/sites.json 包含白名单、seed_urls、部门映射、主题关键词、深度、访问间隔、单文件大小上限和重试次数。种子应来自已核验的官方站点。当前配置提供招生、教学质量报告、学生工作、图书馆、财务和在校学生导航入口；当前结果并非完整主题知识库。

输出位于 data/raw：

- manifest.jsonl：追加式采集历史，含 URL、最终 URL、标题、部门、日期、日期来源、父页面、哈希、本地路径和状态。
- crawl_report.json：最近一次运行的成功、失败、附件、未变化数量和部门分布。
- 哈希命名的 HTML、PDF、DOCX、XLSX 等原始文件。

未知日期保持 null，不使用抓取日期冒充发布日期。附件若继承父页面日期，会标记 date_source=parent_page。文件哈希相同的资源共享本地文件，但来源分别记录。PDF 播放器的 pdfsrc 属性也会被识别。当前不执行网页 JavaScript、OCR，不绕过验证码或认证；旧 DOC/XLS 可下载但暂不在导入脚本支持范围。

检查最近一次报告：

```powershell
Get-Content data/raw/crawl_report.json -Encoding UTF8
```

抽查 manifest 的发布日期、附件 parent_url 和真实文件内容，排除错误提示页。只有确认数据质量后再运行：

```powershell
python scripts/clean_data.py
python scripts/ingest_data.py --clean-manifest data/processed/clean/clean_documents.jsonl
python scripts/check_database.py
python scripts/build_index.py
```

清洗读取每个 URL 最新状态，合并相同正文并保留来源。当前版本已支持按清洗内容幂等导入和文档级事务，完整流程见 docs/cleaning.md；旧版无导入标记的数据库需先备份再重建。

robots.txt 不可读取、重定向或服务器拒绝时，本次不访问该站点并在 manifest 记录 robots_denied，需检查网络和站点策略后再重试。网络错误只记录异常类型，不写入请求凭据。每个资源的请求重试默认两次，流式正文传输中断会记为失败、下一轮重新抓取。
