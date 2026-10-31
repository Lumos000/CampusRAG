# 南京邮电大学离线校园问答系统

本项目用于根据南京邮电大学公开资料回答问题，支持终端交互问答和批量生成测试结果 CSV。请按以下步骤使用。

## 1. 解压代码和资源

准备两个文件：

- 代码包：从 GitHub 下载的源码 ZIP，或随提交材料提供的 `campus-rag-code.zip`。
- 资源包：随提交材料提供的 `campus-rag-resources.zip`，包含模型、LoRA 权重、数据库和索引。

将代码包解压到一个目录，例如 `E:\CampusRAG`。找到 `app.py`，它所在的文件夹就是项目根目录。

把资源包放入项目根目录，在该目录打开 PowerShell，执行：

```powershell
Expand-Archive .\campus-rag-resources.zip -DestinationPath .
```

也可以使用解压软件。解压后应有以下目录结构：

```text
CampusRAG/
├── app.py
├── pyproject.toml
├── config/
├── scripts/
├── src/
├── models/
│   ├── Qwen3-0.6B/
│   ├── bge-small-zh-v1.5/
│   ├── bge-reranker-base/
│   └── adapters/planner_lora/step_300/
└── data/processed/campus_public/
    ├── campus_public.db
    └── index/
```

资源解压完成后，直接进行下面的安装和运行步骤。

## 2. 安装运行环境

以下命令在 Windows 的 Anaconda Prompt 或已配置 Conda 的 PowerShell 中执行。将 `E:\CampusRAG` 替换为实际项目路径：

```powershell
cd E:\CampusRAG
conda create -n campusrag python=3.11
conda activate campusrag
python -m pip install -e ".[lora]"
```

已有 Python 3.11 的 Conda 环境时，激活该环境并执行最后一条安装命令即可。依赖安装需要联网，安装完成后的问答可以断网运行。

程序会自动使用可用的 CUDA 显卡，没有可用显卡时使用 CPU。若需要使用 NVIDIA 显卡，请安装与本机驱动匹配的 CUDA 版 PyTorch。已验证的设备为 RTX 4050 Laptop 6GB；CPU 推理速度和内存占用需按实际设备确认。

## 3. 检查资源

保持 Conda 环境已激活，在项目根目录执行：

```powershell
python scripts/check_database.py
python scripts/check_offline.py
```

检查结果中的 `checks` 和 `ok` 项应为 `true`。如果提示文件缺失，检查资源包是否完整解压到了 `app.py` 所在目录。

## 4. 在终端输入问题

执行：

```powershell
python app.py
```

首次加载模型需要等待一段时间。出现 `问题>` 提示后，输入问题并按回车，例如：

```text
南邮离线校园问答。输入 exit 退出。

问题> 根据2026年本科招生章程，工科类优势学科专业的学费和住宿费预收标准分别是多少？
```

程序会输出答案，有引用时还会显示参考资料。随后可继续输入下一道问题。输入 `exit` 并按回车退出。

以后重新打开终端时，只需执行：

```powershell
conda activate campusrag
cd E:\CampusRAG
python app.py
```

## 5. 批量测试并导出结果

将测试题保存为 `questions.csv`，放在项目根目录。文件采用 UTF-8 编码，第一行为列名 `问题`，之后每行一道题：

```csv
问题
图书馆如何办理续借？
2026年本科生住宿费标准是多少？
```

可使用 Excel 编辑一列题目，第一格填写“问题”，再另存为“CSV UTF-8（逗号分隔）”。如果收到的是 XLSX 或 Word 文件，先整理为上述 CSV 格式。

运行：

```powershell
python scripts/batch_answer.py --input questions.csv --output results.csv
```

等待全部题目处理完成后，项目根目录会生成 `results.csv`。输出采用 UTF-8 with BOM 编码，固定包含三列：

```csv
问题,答案,参考资料
```

正式测试时，提交生成的 `results.csv` 即可。默认使用配套的校园数据库和数值规划 LoRA。

文件也可以放在其他位置，命令中填写完整路径，含空格的路径用双引号包住：

```powershell
python scripts/batch_answer.py --input "D:\测试题\questions.csv" --output "D:\测试结果\results.csv"
```

如需先使用项目附带的 54 道问题试运行，执行：

```powershell
python scripts/batch_answer.py --input data/eval/campus_public/questions.csv --output sample_results.csv
```

附带的参考输出位于 `data/eval/campus_public/answers.csv`。
