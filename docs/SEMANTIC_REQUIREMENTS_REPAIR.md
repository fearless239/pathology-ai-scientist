# 混合要求与标签平滑语义验收修复

## 修改范围

- 对旧合同的 implementation_signals 做只读分类：干预方法、架构、初始化、数据策略、评价指标、基础损失。新合同同时记录 requirement_groups。旧合同内容及哈希不变。
- smoothing_factor 与 label_smoothing 合并为同一干预概念。conv_layers、num_classes、from_scratch、train_subset_fraction、test_accuracy 不再作为需要调用同名函数的干预组件。
- 分类不代表这些要求已通过科学验收：架构、初始化、数据使用和最终评价仍需在对应的模型/数据/评价及研究审核环节验证。本次修复没有建立完整的数据流或模型结构形式化证明器。
- 接受 cnn_architecture、data_loading、classification_metrics、sgd、supervised_training 等标准元数据分类；未知分类首次进入审核即暂停，不自动连续生成错误标签。
- 前置语义检查与导出使用同一 passed 判断，变量名、注释或导入不能单独使缺失方法通过。

## 自定义标签平滑

优先提示模型复用 PyTorch 内置实现。对当前支持的自定义 Module 形式（log_softmax、gather、mean 与 smoothing 属性），静态检查只识别待验证候选，不能认定数值正确。

可信运行器检查实际调用的损失值及 logits 梯度是否与 PyTorch cross_entropy(label_smoothing=...) 一致；挂接输出梯度钩子，结束时要求该损失确实经过反向传播。错误值、错误梯度、从未使用的损失都不能通过。每个损失 Module 的首次可求导调用执行数值检查；这不是任意自定义损失的等价性证明。其他形式仍需审核，不能一律按名称放行。

## 证据与验证

tests/fixtures/semantic_recovery 保存此次真实失败的最终生成代码及原始 intervention 定义，不是简化的单个 label_smoothing 信号。测试同时走 semantic_report、导出语义检查和第 3 阶段预检入口。

已有全量回归通过 414 项，1 项跳过（本机无 torch）。另在已有 PyTorch 镜像中，以禁网、无 GPU、无 API 密钥的 CPU 环境调用同一组回归函数：正确损失通过；错误值、错误梯度、未使用损失均被拒绝。镜像没有 pytest，因此用标准库加载并直接运行测试函数，没有安装依赖或伪造 pytest 运行结果。

没有执行真实任务训练，没有修改 pathtest-001 的合同、checkpoint、节点状态、重试次数或费用上限。之前恢复脚本仅适用于第 2 阶段，不能用于现在的第 3 阶段。后续如需恢复，必须单独准备并核对已完成调参结果；本次未声称整个科研任务已跑通。
