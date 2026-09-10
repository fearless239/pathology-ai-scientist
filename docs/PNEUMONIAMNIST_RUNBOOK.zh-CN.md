# PneumoniaMNIST 前端端到端运行手册

本手册用于创建 `pneumoniatest-001`。PneumoniaMNIST 是胸部 X 光二分类数据，属于生物医学图像
适配验证，不应描述为病理数据集或临床系统。

## 1. 准备数据

从 MedMNIST 唯一官方发行页下载 `pneumoniamnist.npz`：

<https://zenodo.org/records/10519652>

预期文件大小约 4.2 MB，MD5 为 `28209eda62fecd6e6a2d98b1501bb15f`。官方数据统计应为：

| 字段 | 预期值 |
|---|---:|
| 图像 | 28×28 灰度图 |
| 类别 | 2（normal / pneumonia） |
| train | 4,708 |
| validation | 524 |
| test | 624 |

默认把文件放在项目同级目录：

`C:\Users\asd\Desktop\autoresearch\datasets\pneumoniamnist.npz`

数据不进入 Git 仓库。启动脚本会把项目同级的 `datasets` 目录只读挂载到网页容器的
`/datasets`。若要使用其他宿主目录，请先在 WSL 中设置 `PATH_AI_DATASETS_DIR` 为对应的
Linux/WSL 路径。

## 2. 启动前端

双击仓库根目录的 `启动网页.bat`。脚本会通过 WSL 启动 Streamlit，并打开：

<http://127.0.0.1:8501>

如果需要手动启动，在拥有 Docker Engine 的 WSL 环境执行：

```bash
bash scripts/pathmnist.sh web --server.headless true
```

## 3. 创建任务

- 课题 ID：`pneumoniatest-001`
- 数据适配器：`generic`
- 数据集路径：`/datasets/pneumoniamnist.npz`
- 自动拆分随机种子：`7`（官方 split 存在，因此不会重新拆分）
- LLM 预算上限：`10`

研究方向粘贴以下完整文本：

> 在 PneumoniaMNIST 官方划分上，从零训练相同的轻量级 CNN，比较普通交叉熵与仅加入训练集逆频率类别权重的加权交叉熵。以 macro-F1 为主指标，accuracy 和 weighted-F1 为次指标；固定 3 个随机种子进行配对训练，除 loss weighting 外保持模型、数据、优化器、学习率、batch size、epoch 上限和 early stopping 完全一致。不访问额外数据或预训练权重，在所有候选冻结后仅执行一次获批的密封测试，并如实报告正负结果与统计限制。

创建后先核对数据画像：类别数 2、通道数 1、shape 28×28、split 数量与上表一致，并确认研究视图中
没有 test 数组。

## 4. 审批顺序

1. 先让系统运行到研究合同审批点。
2. 核对合同包含 `macro_f1`、3 个 seeds `[0,1,2]`、class-weighted cross-entropy，以及唯一变量原则。
3. 批准研究合同。
4. 勾选允许本机 GPU/Docker 和付费 API，启动实验。
5. baseline 与 proposed 各完成 3 个配对 seed 后，核对固定控制检查通过。
6. 只有全部候选冻结后，勾选并批准唯一一次 sealed test。
7. 测试完成后继续允许付费写作。
8. 到 PDF 阶段再允许 Docker PDF 构建。

不要因验证集或测试集结果为负而修改研究问题、替换 intervention 或重新运行 sealed test。

## 5. 最终验收

- baseline 和 proposed 各有 seeds 0、1、2 的完整证据。
- 两组除 class weighting 外的固定控制完全相同。
- comparison statistics 的 `n` 为 3。
- sealed-test `attempt_count` 为 1。
- 报告包含 mean、std、paired differences 和单独的各类指标。
- `hypothesis_supported`、`review_decision`、`artifact_validation_passed` 分开呈现。
- 英文和中文 PDF 均存在且 PDF QA 通过。
- evidence ZIP 非空，最终任务归档且不能仍为 `interrupted`。
- 总结如实保留 reviewer 的 Accept/Reject 决策，不以归档状态代替学术接收。
