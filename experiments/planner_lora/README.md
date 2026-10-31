# 数值规划 LoRA

本目录保留最终适配器的数据生成、训练和复测代码。正式运行使用 `models/adapters/planner_lora/step_300/` 的配置和权重；已有权重无需重训。

LoRA 只参与数值计划及一次修复，普通文字回答使用基础模型。Python 校验计划、计算并生成引用。配置 `planner_adapter_path` 为 `null` 或使用 `--no-adapter` 可关闭。

## 验证

```powershell
python experiments/planner_lora/verify_production.py --output .tmp/verification
```

使用正式批量入口，关闭和开启适配器各运行 54 题，推理时禁止网络连接。输出三列 CSV、逐题审计和 `comparison_summary.json`。题目是开发回归，不能代表未知问题准确率。`results/production.json` 保存正式批量入口的核对摘要，`results/summary.json` 保存最终训练摘要。

## 重现训练

需要本地基础模型、数据库、索引及支持 BF16 的 CUDA 环境。训练写入 `models/adapters/planner_lora/`，请使用独立项目副本避免覆盖正式权重。`data/eval/planner_lora/` 必须是新工作目录，脚本拒绝覆盖已冻结或已完成的实验。

```powershell
python -m pip install -r experiments/planner_lora/requirements.txt
python experiments/planner_lora/build_data.py
python experiments/planner_lora/prepare.py
$env:CAMPUSRAG_EXPERIMENT_ROOT = (Get-Location).Path
python data/eval/planner_lora/snapshot/experiments/planner_lora/run.py --preflight
python data/eval/planner_lora/snapshot/experiments/planner_lora/run.py
```

固定参数：BF16、r=8、alpha=16、all-linear、学习率 2.5e-5、两轮共 300 次更新、微批量 1、梯度累积 8、最长 6144 tokens。先做最长输入前向与反向预检，通过后再训练。

生成 1200 条训练、240 条验证、240 条测试样本，按资料实例和问法族隔离。验证集比较第 150/300 步，测试不参与选点。`common.py`、`canonical_score.py`、`runtime_eval.py` 提供评分和正式调用链评测；`data_templates.py` 提供生成模板；`scripts/audit_planner_training.py` 检查训练覆盖。

训练保存必要输入快照以支持续跑，运行产物不提交。训练后自动执行真实资料库对照，合成题评分不替代真实问答验证。
