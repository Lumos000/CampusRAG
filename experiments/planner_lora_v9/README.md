# V9 数值规划 LoRA 实验

本目录包含 V9 的数据生成、参考计划评分、训练、断点恢复、完整修复链评测和真实库回归工具。只对数值规划和修复启用 LoRA；普通文本回答仍使用基础模型。正式 `app.py` 和批量回答入口尚未加载适配器。

## 依赖与本地资源

在 Python 3.11 的独立环境中安装项目及实验依赖：

```powershell
python -m pip install -e ".[dev]"
python -m pip install -r experiments/planner_lora_v9/requirements.txt
```

CUDA 版 Torch 请按本机驱动单独安装。已完成实验的配置为 Qwen3-0.6B、BF16、r=8、alpha=16、all-linear、学习率 2.5e-5、两轮共300次更新、微批量1、梯度累积8、最长6144 tokens。训练要求 CUDA；6GB 显存可否运行必须先做本机预检。

以下资源由本地准备，不随代码提交：

- `config/settings.json` 指定的 Qwen、嵌入及重排模型、校园数据库和索引。
- 对照权重 `models/adapters/planner_lora_v8/step_300/`。V9 的三方对照及预检需要它。
- 已训练 V9 的权重位于 `models/adapters/planner_lora_v9/step_300/`；本仓库不包含权重。

本目录自带继承的数据模板、评分器及36题回归标签，不依赖 V1—V8 的实验代码目录。真实回归还使用仓库已有的 `data/eval/campus_public/verified_release.json`，并绑定该次校园数据库的表、字段和文档ID；换库后需要重新审查标签与引用，不能直接报告原分数。

规范化版本已升级到2，旧库须先备份，再运行：

```powershell
python scripts/normalize_tables.py --database <数据库路径>
```

## 新实验顺序

以下命令均从仓库根目录执行。已存在的实验数据和冻结协议不会被覆盖；保留已完成实验后，使用独立工作副本开展新实验。整理代码没有改写本机已完成的冻结快照，也没有重新训练。

```powershell
# 小批审查；输出必须是尚不存在的目录
python experiments/planner_lora_v9/build_data.py --pilot --output .tmp/v9-pilot

# 独立生成1200/240/240条及可执行参考数据库
python experiments/planner_lora_v9/build_data.py

# 先测同一新接口下的V8，避免把代码修复算作训练收益
python experiments/planner_lora_v9/run_comparison.py --output data/eval/planner_lora_v9/interface_baseline --adapter models/adapters/planner_lora_v8/step_300
python experiments/planner_lora_v9/review_comparison.py --directory data/eval/planner_lora_v9/interface_baseline

# 审查全部参考计划、覆盖、长度、分集隔离并冻结输入
python experiments/planner_lora_v9/prepare.py

# 正式训练前，使用冻结脚本做最长输入前向/反向及显存预检
$env:CAMPUSRAG_EXPERIMENT_ROOT = (Get-Location).Path
python data/eval/planner_lora_v9/snapshot/experiments/planner_lora_v9/run.py --preflight

# Windows 后台训练，随后自动选点、测试和真实库回归
python experiments/planner_lora_v9/launch.py
```

`launch.py` 仅用于 Windows；其他系统可用自身进程管理工具运行上述冻结的 `run.py`。恢复同一次中断实验使用相同命令；脚本验证冻结输入哈希及断点状态。已经完成的实验不能重新启动。

查看进度无需持续监控：`Get-Content data/eval/planner_lora_v9/status.json`。验证集只比较150/300步并选择检查点，测试集不参与选择。合成评测从模型首轮计划出发，最多使用一次真实反馈修复；评测不提供教师错误计划。

## 已完成结果与限制

`results/summary.json` 摘录已完成实验记录，整理代码后没有重跑GPU评测，不把下列分数作为新一次运行结果：

| 同一运行接口 | 基础模型 | V8 | V9 step_300 |
| --- | ---: | ---: | ---: |
| 合成验证，240题 | 65 | 145 | 175 |
| 合成测试，240题 | 47 | 136 | 150 |
| 当前真实库开发回归，54题 | 49 | 53 | 53 |

V9真实回归相比基础模型改善4题、没有原正确题退步；相比V8改善NEW06、退步C03。合成测试相比基础模型改善108题、退步5题；相比V8改善31题、退步17题。V9合成测试仍有两个错误数值和一个数值正确但来源证据不足的可执行输出，因此严格门槛未通过，适配器默认关闭。这里的“错误可执行输出”包含语义或证据不合格，不等同于算术错误。

真实54题是已暴露的开发回归，不能宣称是新的独立盲测。任务书4题三种模式都是4/4，其中3题使用确定性工具、1题调用规划模型，不能把全部示例题成功归因于LoRA。

全套代码回归：`python -m pytest -q`。模型、训练状态、生成数据、数据库、索引和大体积逐题日志均不属于本次代码提交。

本次整理已在“当前已提交项目＋本次27个文件”的独立副本中验证：333项测试通过；重新生成的1680条数据与原实验逐条一致；1680个参考计划通过来源、单位与有序操作数检查；原三组测试的可执行最终计划判定及两组真实库回归汇总保持一致。未重新训练或重跑GPU推理。
