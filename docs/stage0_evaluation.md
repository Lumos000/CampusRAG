# 阶段 0：冻结与诊断

原实现为 `a44fed7848cc8df4ffbc94df2f57e4a65a9a25e8`；原 36 题金标、运行结果、逐题人工语义复核与失败分析保存在 `2e55d37`。不修改这些金标或覆盖 `independent_v1/run1`。原结果为严格语义正确 19/36、计算正确 1/8，12 题开发回归为 12/12。不同集合、不同评分口径不能直接当作能力升降。

本阶段只增加观测，不修改检索、提示、Facts 抽取、字段选择、执行或回答规则。普通回答与快捷计算也记录最终引用。`RAGService.last_diagnostics` 每次请求重置，记录检索证据 ID、实际提供的 Schema/Facts、首次与修复规划原文、错误反馈、绑定后的计划及字段、执行明细与最终引用。`syntax_ok`、`execution_ok` 与 `semantic_correct` 分开；最后一项保持 `null`，须在评测中按答案和证据复核，不能由程序执行成功推断。

留出集 v2 在回答逻辑修改前冻结于 `data/eval/independent_v2/`：16 题，包括数值 8 题、普通文字 4 题、时效 2 题、拒答 2 题。源文引文经原始 PDF/HTML 独立抽取校验并记录哈希。阶段 1～3 禁止运行或根据其结果调试。同一助手编题评分且若干题同源，这只是预先固定的留出集，不是外部双盲基准。原文抽取与数据库个别措辞存在差异，金标仍以原始资料为准；后续失败须区分资料质量和推理错误。

阶段复跑使用 `scripts/evaluate_stage.py`，为当前源码生成新快照，继续使用原 36 题，不绕过快照校验。每阶段输出独立目录，已有目录不能重新准备、已有结果不能覆盖；`--resume` 仅补完尚未运行的题。`scripts/compare_stage.py` 仅在答案、实际引用和计算对象完全一致时复用原语义评审，任何变化都标记待复核，不自动判对。

运行命令：

```powershell
python -m pytest -q
python scripts/evaluate_stage.py --stage stage0 --prepare
python scripts/evaluate_stage.py --stage stage0
python scripts/evaluate_numeric.py --output data/eval/stages/stage0/numeric_results.json
python scripts/compare_stage.py --directory data/eval/stages/stage0
```

真实模型评测串行运行，复用现有数据库和索引，无需重新导入资料。三列提交 CSV 仍为“问题、答案、参考资料”；诊断信息另存 JSON，不加入提交列。

自动测试：79 项通过（包括一次修复记录、请求间清理、变化答案不得复用评审）。真实 Qwen 模型串行复跑结果：12 题开发回归 12/12；36 题答案、实际引用及计算对象全部与原记录一致，严格语义正确率仍为 19/36（52.78%），数值正确率 1/8（12.5%）。无新增运行异常。逐题对照见 `data/eval/stages/stage0/comparison.json`；运行日志、语义评审及汇总见同目录下的 `run1/`。

计算诊断复现：C01/C04 无可用 Facts 且未调用规划；C02 两次引用不存在的表格；C03 两次字段绑定失败；C05 一次规划成功；C06 首次对象绑定失败、修复 JSON 不完整；C07/C08 分类字段重复匹配被当作不可修复错误。下一阶段先改表示层；这些执行状态不等同于答案语义评分。
