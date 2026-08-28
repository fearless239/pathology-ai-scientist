# 指标语义与产物保护修复

本次针对 pathtest-001 暴露的项目级问题修复，不修改该任务的合同、历史结果或 checkpoint，不自动恢复训练。

## 指标契约

新实验 manifest 使用三个独立概念：

```json
{
  "primary_metric": "accuracy",
  "selection_metric": "accuracy",
  "checkpoint_selection": {"metric": "validation_loss", "mode": "min"},
  "early_stopping": {
    "enabled": true, "monitor": "validation_loss", "mode": "min",
    "patience": 5, "min_delta": 0.0
  }
}
```

`selection_metric` 是新记录中研究主指标的兼容字段。候选学习率按主指标比较；候选内部按 checkpoint_selection 选轮次；早停策略不再被误当作主指标。调参保持基线 checkpoint 和早停策略不变。

每个 tuning_evidence 候选需记录 `selected_epoch`。其 `validation_metric` 是该轮保存权重对应的研究主指标，不一定是训练历史中的最高准确率。验收会检查选中轮次满足 checkpoint_selection、候选间选中最高主指标及最终预测对应的可信指标。

旧 manifest 的 `selection_metric=validation_loss` 只读解释为 loss/min 权重选择；不会改写历史文件。新候选可使用主指标 accuracy 和相同 loss/min 权重选择策略。`validation_metric` 等无确切含义的旧别名明确阻塞迁移，不猜测为 accuracy。

## 错误与重跑

成功执行产生新结果后，先将代码、权重、原始结果及训练记录发布到独立 `experiment_logs/raw_executions` 目录，再做验收。raw_receipt 标记为 unvalidated，不能直接用于冻结或发表；也不能仅凭文件存在认定可恢复。

manifest/键缺失错误触发 ARTIFACT_REVIEW_REQUIRED，Manager 在下一次训练前暂停，避免为修 JSON 重训。后续需核实原始记录及权重，输出新的、来源可追溯的验证结果；目前不会自动改写旧证据或自动批准迁移。代码执行错误仍按上游节点调试路径处理。

失败摘要现在保留被标记为 buggy 的节点原因，不再因过滤全部失败节点而只输出 No verified tuning candidate。

## 数据分辨率

允许使用现有 64×64 源数据，在训练、验证、推理时一致缩放到 28×28。无需下载另一份数据。任务生成提示及论文数据描述明确区分源 image_shape 和模型 input_resolutions，要求报告缩放和插值，不能写成源数据原本就是 28×28。历史批准文件未被改写。

## 验证与边界

本轮全量结果：407 passed、1 skipped，Ruff 通过；14 条警告来自第三方 matplotlib/pyparsing 弃用接口。

- 新增真实失败形态的隔离回归：旧基线 loss/min，候选主指标 accuracy，保持 loss/min 选择权重。
- 验证准确率最高轮次与损失最低轮次不同的情况、错误轮次拒绝、含糊别名诊断、原始权重留存及暂停保护。
- 原生 Manager 串联矩阵改用显式分离指标契约，覆盖四类指标和可选调参/消融。
- 未做付费 GPU/LLM 验收，未重新启动 pathtest-001。旧任务已耗尽的迭代窗口不会被本修复偷偷扩大；恢复前需审核现有证据并明确批准恢复策略。
