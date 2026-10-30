# 数值规划 LoRA 说明

正式推理采用 Qwen3-0.6B 与本地数值规划 LoRA。配置 `planner_adapter_path` 指向 `models/adapters/planner_lora/step_300`，交互入口 `app.py` 与批量入口 `scripts/batch_answer.py` 均支持 `--no-adapter` 关闭。普通文本回答关闭适配器，数值规划和修复启用，计算与引用仍由程序校验。

实验代码集中于 `experiments/planner_lora/`，包含数据模板、参考评分、完整修复链评测、训练恢复及正式入口对照，不需要其他版本代码或权重。安装、推理、复现及关闭方式见该目录 `README.md`。

已完成实验：合成验证65→175/240，合成测试47→150/240，真实库开发回归49→53/54。真实回归无原正确题退步；合成测试退步5题，存在两个错误数值和一个来源不合格的可执行输出，严格零退步门槛未通过。成绩摘要见 `experiments/planner_lora/results/summary.json`，不能将开发回归宣称为新盲测。

正式入口效果检查由 `experiments/planner_lora/verify_production.py` 执行，分别关闭／开启适配器，输出两份CSV、审计记录和逐题对照。新运行结果保存在 `data/eval/planner_lora/production/`，与历史训练摘要区分。

仅在明确表达中提取数值、单位、原文位置和总量／已完成／剩余／计划角色。剩余量减法必须扣除相应已完成量；合法零余量不被禁止。周期冲突反馈指出具体事实与乘数单位，修复时重新检查完整计划；证据不足仍拒答，不猜数字或换算。

模型权重、数据库、索引、训练状态及大体积日志保存在本地，不随代码上传。

2026-10-31正式批量入口实测已完成：关闭／开启适配器各54题，正确数49→53，无原正确题退步、无新增错误数值，网络访问尝试为0。15题实际使用规划适配器，其余39题答案和引用一致。可提交的结果及输入哈希见 `experiments/planner_lora/results/production.json`；逐题CSV和执行审计保存在本地 `data/eval/planner_lora/production_20261031/`。剩余C03问题选择了贷款下限事实，修复未纠正，被拒答。
