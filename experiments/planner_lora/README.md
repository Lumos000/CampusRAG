# 数值规划 LoRA

本目录保留最终数值规划适配器的数据生成、训练、评分及评测代码。模型采用本地 Qwen3-0.6B，只对数值计划生成和一次修复启用 LoRA；普通文本回答使用基础模型，数值运算仍由 Python 执行并校验。

## 正式推理

在独立 Python 3.11 环境中安装项目及适配器依赖：

```powershell
python -m pip install -e ".[dev,lora]"
```

本地资源包括配置指定的基础模型、嵌入模型、重排模型、数据库与索引，以及最终权重：

```text
models/adapters/planner_lora/step_300/adapter_config.json
models/adapters/planner_lora/step_300/adapter_model.safetensors
```

权重和数据库不随代码上传。配置中的 `planner_adapter_path` 指向上述目录；设为 `null` 可长期关闭，`--no-adapter` 可仅关闭本次推理。缺少权重或 PEFT 时明确报错，不静默改用其他模型。

```powershell
python scripts/check_offline.py
python app.py
python app.py --no-adapter
python scripts/batch_answer.py --input questions.csv --output results.csv
python scripts/batch_answer.py --input questions.csv --output base_results.csv --no-adapter
```

交互与批量入口使用相同的生成器和规划分流。批量输出仍为“问题、答案、参考资料”三列；另加 `--audit-output audit.json` 可保留计算过程、适配器状态及执行诊断。适配器不会参与普通文本句段选择或确定性数值工具。

## 已完成实验

| 同一接口下的评测 | 未微调 | 最终适配器 |
| --- | ---: | ---: |
| 合成验证，240题 | 65 | 175 |
| 合成测试，240题 | 47 | 150 |
| 当前真实库开发回归，54题 | 49 | 53 |

这些分数来自已完成的训练实验，摘要见 `results/summary.json`。真实54题是已暴露的开发回归，不是新的盲测。合成测试相比基础模型改善108题、退步5题，仍有两个错误数值和一个数值正确但来源不足的可执行输出；严格零退步门槛未通过。任务书4题两种模式均为4/4，其中3题走确定性工具，不能把全部成功归因于微调。

重新验证正式批量入口：

```powershell
python experiments/planner_lora/verify_production.py
```

该命令使用真实批量入口关闭／开启适配器各运行54题，阻断网络，校验输入文件哈希，保存两份CSV、逐题审计及 `comparison_summary.json`。默认输出 `data/eval/planner_lora/production/`，必须是新目录；也可用 `--output` 指定其他新目录。评测标签绑定当前校园数据库的文档、表和字段ID，换库后需要重新核对。

正式批量入口已于2026-10-31完成关闭／开启适配器各54题的实测：49/54（90.7%）→53/54（98.1%），新增正确4题、原正确题退步0题、返回错误数值均为0。15题实际调用规划适配器，其余39题答案和引用完全一致，全程网络访问尝试为0。整批耗时含模型加载分别约122秒和160秒。剩余C03贷款上限差额题误选下限事实，修复未改正，被校验拦截并拒答。结果与输入哈希见 `results/production.json`；逐题CSV和审计保存在本地 `data/eval/planner_lora/production_20261031/`。代码测试340项通过，适配器资源、数据库及索引检查通过。真实回归通过不代表既有合成测试中的问题已经解决。

## 重现训练

复用最终模型无需重新训练。以下流程用于新实验，不覆盖已有结果，不依赖其他适配器：

```powershell
python -m pip install -r experiments/planner_lora/requirements.txt
python experiments/planner_lora/build_data.py --pilot --output .tmp/planner-pilot
python experiments/planner_lora/build_data.py
python experiments/planner_lora/prepare.py
$env:CAMPUSRAG_EXPERIMENT_ROOT = (Get-Location).Path
python data/eval/planner_lora/snapshot/experiments/planner_lora/run.py --preflight
python experiments/planner_lora/launch.py
```

训练使用 BF16、r=8、alpha=16、all-linear、学习率2.5e-5、两轮300次更新、微批量1、梯度累积8、最长6144 tokens。要求 CUDA，先做本机最长输入前向／反向预检。训练和最终推理使用不同的输出目录或工作副本，避免覆盖正在使用的权重。

数据生成1200条训练、240条验证、240条测试，按资料实例和问法族隔离；保留角色竞争、合法零余量、长输入及多错误修复。验证集比较150/300步选点，测试不参与选择。评测从模型自己的首轮计划出发，最多一次真实反馈修复，不提供教师错误计划。

`launch.py` 用于 Windows 后台运行，其他系统可用自己的进程管理工具执行冻结的 `run.py`。恢复同一次中断实验使用原命令；输入哈希变化时拒绝恢复。已完成实验不能重新启动。

规范化版本为2，旧库先备份，再运行 `python scripts/normalize_tables.py --database <数据库路径>`；正式推理只读数据库。

代码测试：`python -m pytest -q`。代码整理保留了原实验记录，未通过重命名改写历史分数。
