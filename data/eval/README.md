# 验收数据

本目录只保留复测输入及一份最终批量输出。自动生成的日志、审计、训练样本和对比结果不纳入代码提交。

| 文件 | 用途 |
| --- | --- |
| `campus_public/questions.csv` | 54 道固定问题，包含任务书四例 |
| `campus_public/answers.csv` | 精简代码与资源包在干净目录恢复后的真实批量输出 |
| `campus_public/verified_release.json` | 48 道校园题的人工核对参考，供适配器验证使用 |
| `college_dataset_v3/cases.json` | 按来源内容键标注的检索题，不依赖分块 UUID |

54 题由 48 道校园题加 6 道学分学时计算组成，均已用于开发回归，不是盲测。参考文件的旧运行摘要不作为当前效果声明，当前结果通过验证入口获取。

随附输出为 53/54，任务书四例通过，贷款上限差额题拒答。实测从代码包与离线资源包解压出的干净目录运行，基础模型为 49/54，开启 LoRA 后为 53/54，无原正确题退步，无错误数值输出，推理期间网络访问尝试为 0。核对摘要为 `experiments/planner_lora/results/production.json`，记录运行源码、配置及资源哈希。

```powershell
python scripts/batch_answer.py --input data/eval/campus_public/questions.csv --output results.csv
python experiments/planner_lora/verify_production.py --output .tmp/verification
```

第二个命令使用正式批量入口分别关闭、开启 LoRA，检查数值、单位、来源和网络调用，输出两份 CSV 和 `comparison_summary.json`。参考答案与指定数据库绑定；更换资料或改变数据库 ID 后需重新核对。文字答案与参考不同时，脚本标记为待复核。

检索题集用于两份数据库的四模式对照：

```powershell
python scripts/compare_college_retrieval.py --baseline-config baseline.json --candidate-config config/settings.json --cases data/eval/college_dataset_v3/cases.json --output .tmp/retrieval_comparison
```

两份配置必须指定不同数据库和索引，使用相同嵌入及重排模型。报告按题目来源分组显示来源命中率及 MRR；来源命中不等于答案正确。
