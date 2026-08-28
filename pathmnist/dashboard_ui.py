from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


APP_CSS = """
<style>
  :root {
    --ps-bg: #101110;
    --ps-panel: #171817;
    --ps-panel-raised: #1d1f1d;
    --ps-panel-hover: #242624;
    --ps-border: rgba(255, 255, 255, 0.09);
    --ps-border-strong: rgba(255, 255, 255, 0.16);
    --ps-text: #f2f3ef;
    --ps-muted: #a0a49e;
    --ps-faint: #747871;
    --ps-accent: #b9f27c;
    --ps-accent-strong: #91df55;
    --ps-purple: #b9a7ff;
    --ps-warning: #f1bd70;
    --ps-danger: #ff8f83;
  }

  html, body {
    font-family: Inter, "SF Pro Display", "Segoe UI", "Microsoft YaHei", sans-serif;
  }

  [data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded" !important;
    font-style: normal;
    font-weight: normal;
    letter-spacing: normal;
    text-transform: none;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
  }

  .stApp,
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 76% -10%, rgba(137, 221, 85, 0.07), transparent 30rem),
      var(--ps-bg);
    color: var(--ps-text);
  }

  [data-testid="stHeader"] {
    background: rgba(16, 17, 16, 0.86);
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
  }

  [data-testid="stToolbar"] {
    display: flex;
  }

  [data-testid="stToolbarActions"],
  [data-testid="stAppDeployButton"],
  [data-testid="stMainMenu"] {
    display: none;
  }

  /* Keep Streamlit's native collapse/reopen controls available.  Hiding every
     header button also hides the only way to restore a collapsed sidebar. */
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapseButton"] button,
  [data-testid="stSidebarCollapsedControl"] button,
  [data-testid="stExpandSidebarButton"] button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  }

  [data-testid="stSidebar"] {
    background:
      radial-gradient(circle at 18% 3%, rgba(110, 168, 255, .07), transparent 13rem),
      linear-gradient(180deg, rgba(30, 30, 32, .98), rgba(18, 18, 20, .995));
    border-right: 1px solid var(--ps-border);
    box-shadow: inset -1px 0 rgba(255, 255, 255, .025), 18px 0 48px rgba(0, 0, 0, .12);
    backdrop-filter: blur(28px) saturate(115%);
  }

  [data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
  }

  [data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start;
    min-height: 2.7rem;
  }

  [data-testid="stSidebar"] .stButton > button p {
    color: inherit !important;
  }

  [data-testid="stSidebar"] .st-key-sidebar-create button[kind="primary"] {
    color: #f5f5f7 !important;
    background: linear-gradient(180deg, rgba(61, 61, 65, .96), rgba(38, 38, 41, .98)) !important;
    border-color: rgba(255, 255, 255, .14) !important;
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, .12),
      0 10px 30px rgba(0, 0, 0, .24) !important;
  }

  [data-testid="stSidebar"] .st-key-sidebar-create button[kind="primary"]:hover {
    color: white !important;
    background: linear-gradient(180deg, rgba(73, 73, 77, .98), rgba(45, 45, 48, .99)) !important;
    border-color: rgba(255, 255, 255, .22) !important;
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, .17),
      0 12px 34px rgba(0, 0, 0, .3) !important;
  }

  [data-testid="stSidebar"] .st-key-sidebar-workspace button {
    color: #e8e8ed !important;
    background: rgba(255, 255, 255, .035) !important;
    border-color: rgba(255, 255, 255, .075) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .025);
  }

  [data-testid="stSidebar"] .st-key-sidebar-workspace button:hover {
    color: white !important;
    background: rgba(255, 255, 255, .065) !important;
    border-color: rgba(255, 255, 255, .12) !important;
  }

  [data-testid="stSidebar"] [class*="st-key-sidebar-task-"] button {
    color: #e8e8ed !important;
    background: rgba(255, 255, 255, .025) !important;
    border-color: rgba(255, 255, 255, .06) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .018);
  }

  [data-testid="stSidebar"] [class*="st-key-sidebar-task-"] button:hover {
    color: white !important;
    background: rgba(255, 255, 255, .06) !important;
    border-color: rgba(255, 255, 255, .11) !important;
  }

  [data-testid="stSidebar"] [class*="st-key-sidebar-task-"] button[kind="primary"] {
    color: white !important;
    background:
      linear-gradient(90deg, rgba(120, 179, 255, .09), rgba(255, 255, 255, .075)) !important;
    border-color: rgba(255, 255, 255, .14) !important;
    box-shadow:
      inset 2px 0 0 #86bfff,
      inset 0 1px 0 rgba(255, 255, 255, .08),
      0 8px 24px rgba(0, 0, 0, .14) !important;
  }

  .block-container {
    max-width: 1220px;
    padding-top: 2.2rem;
    padding-bottom: 6rem;
  }

  h1, h2, h3, h4, h5, h6,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stCaptionContainer"] {
    color: var(--ps-text);
  }

  h1 {
    letter-spacing: -0.045em;
    font-size: clamp(2rem, 4vw, 3.4rem);
    line-height: 1.05;
  }

  h2, h3 {
    letter-spacing: -0.025em;
  }

  [data-testid="stCaptionContainer"],
  .stCaption {
    color: var(--ps-muted) !important;
  }

  .stButton > button,
  .stDownloadButton > button,
  [data-testid="stFormSubmitButton"] > button {
    color: var(--ps-text);
    background: var(--ps-panel-raised);
    border: 1px solid var(--ps-border);
    border-radius: 13px;
    min-height: 2.75rem;
    font-weight: 650;
    transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
  }

  .stButton > button:hover,
  .stDownloadButton > button:hover,
  [data-testid="stFormSubmitButton"] > button:hover {
    color: white;
    background: var(--ps-panel-hover);
    border-color: var(--ps-border-strong);
    transform: translateY(-1px);
  }

  .stButton > button[kind="primary"],
  [data-testid="stFormSubmitButton"] > button[kind="primary"] {
    color: #11160d;
    background: var(--ps-accent);
    border-color: var(--ps-accent);
    box-shadow: 0 8px 28px rgba(145, 223, 85, 0.12);
  }

  .stButton > button[kind="primary"]:hover,
  [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    color: #0d120a;
    background: var(--ps-accent-strong);
    border-color: var(--ps-accent-strong);
  }

  .stButton > button:disabled,
  [data-testid="stFormSubmitButton"] > button:disabled {
    color: var(--ps-faint);
    background: #191a19;
    border-color: rgba(255, 255, 255, 0.05);
  }

  div[data-testid="stMetric"] {
    min-height: 7.3rem;
    padding: 1rem 1.05rem;
    background: linear-gradient(145deg, #1c1e1c, #181918);
    border: 1px solid var(--ps-border);
    border-radius: 17px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
  }

  div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--ps-muted);
  }

  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--ps-text);
    font-size: 1.32rem;
  }

  div[data-testid="stForm"],
  [data-testid="stExpander"],
  [data-testid="stVerticalBlockBorderWrapper"] {
    color: var(--ps-text);
    background: rgba(28, 30, 28, 0.78);
    border-color: var(--ps-border) !important;
    border-radius: 18px;
  }

  div[data-baseweb="input"] > div,
  div[data-baseweb="textarea"] > div,
  div[data-baseweb="select"] > div,
  [data-testid="stNumberInput"] > div > div {
    color: var(--ps-text);
    background: #191b19;
    border-color: var(--ps-border);
    border-radius: 13px;
  }

  input, textarea {
    color: var(--ps-text) !important;
    caret-color: var(--ps-accent);
  }

  input::placeholder, textarea::placeholder {
    color: #71756f !important;
  }

  [data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, var(--ps-accent-strong), var(--ps-purple));
  }

  [data-testid="stDataFrame"],
  [data-testid="stJson"] {
    border: 1px solid var(--ps-border);
    border-radius: 15px;
    overflow: hidden;
  }

  [data-testid="stAlert"] {
    color: var(--ps-text);
    background: #1c1e1c;
    border: 1px solid var(--ps-border);
    border-radius: 14px;
  }

  .sidebar-brand {
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: .5rem .25rem 1rem;
  }

  .sidebar-mark {
    display: grid;
    place-items: center;
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 50%;
    color: #161617;
    background: linear-gradient(145deg, #f5f5f7 5%, #b7c2d0 58%, #82b7f5 120%);
    font-weight: 900;
    box-shadow:
      inset 0 1px 1px rgba(255, 255, 255, .9),
      0 0 0 4px rgba(134, 191, 255, .055),
      0 8px 22px rgba(0, 0, 0, .24);
  }

  .sidebar-brand strong {
    display: block;
    color: var(--ps-text) !important;
    font-size: 1rem;
  }

  .sidebar-brand span,
  .sidebar-section,
  .sidebar-task-meta {
    color: var(--ps-muted) !important;
    font-size: .76rem;
  }

  .sidebar-section {
    margin: 1rem 0 .45rem;
    letter-spacing: .08em;
    text-transform: uppercase;
  }

  .sidebar-task-meta {
    margin: -.42rem .65rem .55rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .page-intro {
    margin: .8rem 0 1.8rem;
  }

  .page-eyebrow {
    color: var(--ps-accent);
    font-size: .76rem;
    font-weight: 750;
    letter-spacing: .12em;
    text-transform: uppercase;
  }

  .page-intro h1 {
    margin: .55rem 0 .65rem;
  }

  .page-intro p {
    max-width: 760px;
    margin: 0;
    color: var(--ps-muted);
    font-size: 1.02rem;
    line-height: 1.65;
  }

  .task-header {
    padding: 1.35rem 1.45rem;
    margin-bottom: 1.15rem;
    background: linear-gradient(135deg, rgba(34, 37, 34, .96), rgba(24, 26, 24, .96));
    border: 1px solid var(--ps-border);
    border-radius: 21px;
    box-shadow: 0 18px 45px rgba(0, 0, 0, .15);
  }

  .task-header-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
  }

  .task-header h1 {
    margin: .3rem 0 .55rem;
    font-size: clamp(1.65rem, 3vw, 2.45rem);
  }

  .task-header p {
    max-width: 850px;
    margin: 0;
    color: var(--ps-muted);
    line-height: 1.65;
  }

  .status-pill {
    flex: 0 0 auto;
    padding: .42rem .72rem;
    border: 1px solid rgba(185, 242, 124, .2);
    border-radius: 999px;
    color: var(--ps-accent);
    background: rgba(185, 242, 124, .08);
    font-size: .78rem;
    font-weight: 750;
  }

  .status-pill.warning {
    color: var(--ps-warning);
    border-color: rgba(241, 189, 112, .22);
    background: rgba(241, 189, 112, .08);
  }

  .status-pill.danger {
    color: var(--ps-danger);
    border-color: rgba(255, 143, 131, .22);
    background: rgba(255, 143, 131, .08);
  }

  .ready, .not-ready {
    min-height: 5rem;
    padding: .8rem 1rem;
    border-radius: 14px;
    border: 1px solid var(--ps-border);
    background: #191b19;
  }

  .ready { color: var(--ps-accent); }
  .not-ready { color: var(--ps-warning); }

  .stage {
    padding: .8rem 1rem;
    margin: .65rem 0 1rem;
    color: #d8dbd4;
    background: rgba(255, 255, 255, .025);
    border-left: 2px solid var(--ps-purple);
    border-radius: 0 12px 12px 0;
  }

  [class*="st-key-preset-"] {
    min-height: 12.6rem;
    background:
      radial-gradient(circle at 95% 5%, rgba(185, 167, 255, .13), transparent 8rem),
      linear-gradient(145deg, #1c1e1c, #181918) !important;
  }

  [class*="st-key-preset-"] h3 {
    margin-top: .15rem;
    font-size: 1.1rem;
  }

  [class*="st-key-preset-"] p {
    color: var(--ps-muted);
    font-size: .88rem;
    line-height: 1.55;
  }

  .artifact-placeholder {
    min-height: 12rem;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 1.15rem;
    background:
      linear-gradient(180deg, transparent 18%, rgba(16, 17, 16, .5)),
      radial-gradient(circle at 72% 28%, rgba(185, 167, 255, .2), transparent 25%),
      radial-gradient(circle at 34% 42%, rgba(185, 242, 124, .12), transparent 22%),
      #191b19;
    border: 1px solid var(--ps-border);
    border-radius: 17px;
  }

  .artifact-placeholder strong { color: var(--ps-text); }
  .artifact-placeholder span { color: var(--ps-muted); font-size: .84rem; }

  [data-testid="stImage"] img {
    border: 1px solid var(--ps-border);
    border-radius: 16px;
    background: white;
  }

  .artifact-file {
    min-height: 5.9rem;
    padding: .9rem 1rem;
    margin-bottom: .55rem;
    background: #191b19;
    border: 1px solid var(--ps-border);
    border-radius: 14px;
  }

  .artifact-file strong { display: block; color: var(--ps-text); }
  .artifact-file span { color: var(--ps-muted); font-size: .8rem; }

  .st-key-task-action-bar {
    position: sticky;
    bottom: 1rem;
    z-index: 50;
    margin: 1rem 0;
    padding: .85rem !important;
    background: rgba(28, 30, 28, .94) !important;
    border: 1px solid var(--ps-border-strong) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 50px rgba(0, 0, 0, .38);
    backdrop-filter: blur(18px);
  }

  .empty-state {
    min-height: 24rem;
    display: grid;
    place-items: center;
    text-align: center;
    padding: 3rem;
    border: 1px dashed var(--ps-border-strong);
    border-radius: 22px;
    background: rgba(255, 255, 255, .018);
  }

  .empty-state h2 { margin-bottom: .35rem; }
  .empty-state p { max-width: 520px; color: var(--ps-muted); }

  @media (max-width: 900px) {
    .block-container { padding: 1.5rem 1rem 5rem; }
    .task-header-row { display: block; }
    .status-pill { display: inline-block; margin-top: .85rem; }
    .st-key-task-action-bar { bottom: .5rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .75rem; }
    [data-testid="stColumn"] {
      min-width: calc(50% - .75rem) !important;
      flex: 1 1 calc(50% - .75rem) !important;
    }
  }

  @media (max-width: 600px) {
    [data-testid="stColumn"] {
      min-width: 100% !important;
      flex-basis: 100% !important;
    }
  }
</style>
"""


@dataclass(frozen=True)
class ResearchPreset:
    slug: str
    title: str
    description: str
    prompt: str


RESEARCH_PRESETS = (
    ResearchPreset(
        slug="robustness",
        title="稳健泛化研究",
        description="识别染色、压缩与中心差异造成的捷径学习，并验证更稳健的形态表征。",
        prompt=(
            "研究 PathMNIST 分类模型对染色、颜色和压缩伪影等捷径特征的依赖，"
            "设计形态学一致性训练策略，并评估其对跨分布泛化与预测可靠性的影响。"
        ),
    ),
    ResearchPreset(
        slug="interpretability",
        title="可解释性与不确定性",
        description="联合分析分类性能、低置信度样本与模型关注区域是否符合病理形态。",
        prompt=(
            "基于 PathMNIST 研究病理图像分类的可解释性与不确定性估计，重点分析"
            "类间形态相似样本、低置信度预测和模型关注区域，并建立可靠的风险提示基线。"
        ),
    ),
    ResearchPreset(
        slug="hard-cases",
        title="难例与类别混淆",
        description="从混淆矩阵发现最难类别对，用难例挖掘验证有针对性的性能提升。",
        prompt=(
            "面向 PathMNIST 易混淆组织类别开展难例挖掘研究，以混淆矩阵定位关键类别对，"
            "比较置信度采样与监督对比学习对类别对 F1 和宏平均 F1 的影响。"
        ),
    ),
)


@dataclass(frozen=True)
class TaskSummary:
    task_id: str
    research_direction: str
    completed_stage: str
    updated_at: str


class TaskDeletionError(RuntimeError):
    """A research task cannot be deleted safely from the dashboard."""


def delete_research_task(state_root: Path, task_id: str) -> None:
    """Delete one stopped research task without allowing path traversal."""
    root = state_root.resolve()
    if not task_id or Path(task_id).name != task_id:
        raise TaskDeletionError("Invalid task ID")
    task_root = (root / task_id).resolve()
    if task_root.parent != root or task_root.is_symlink():
        raise TaskDeletionError("Task path escapes the workflow state directory")
    task_path = task_root / "task.json"
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskDeletionError("Task is missing or invalid") from exc
    if task.get("schema_version") != 2:
        raise TaskDeletionError("Only current research-workflow tasks can be deleted here")
    run_path = task_root / "web_run.json"
    if run_path.is_file():
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TaskDeletionError("Task run status is invalid") from exc
        if run.get("state") == "running":
            raise TaskDeletionError("Stop or wait for the running task before deleting it")
    shutil.rmtree(task_root)


@dataclass(frozen=True)
class ArtifactItem:
    path: Path
    kind: str
    label: str
    size_bytes: int


def list_research_task_summaries(state_root: Path) -> list[TaskSummary]:
    summaries: list[TaskSummary] = []
    if not state_root.is_dir():
        return summaries
    for task_path in state_root.glob("*/task.json"):
        try:
            value = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("schema_version") != 2:
            continue
        summaries.append(
            TaskSummary(
                task_id=task_path.parent.name,
                research_direction=str(value.get("research_direction", "")),
                completed_stage=str(value.get("completed_stage", "")),
                updated_at=str(value.get("updated_at", "")),
            )
        )
    return sorted(summaries, key=lambda item: (item.updated_at, item.task_id), reverse=True)


# Pre-release compatibility aliases for existing local imports.
delete_v2_task = delete_research_task
list_v2_task_summaries = list_research_task_summaries


def discover_artifacts(task_root: Path) -> tuple[list[ArtifactItem], list[ArtifactItem]]:
    figure_root = task_root / "paper" / "figures_generated" / "figures"
    preferred = {
        "test_metrics.png": 0,
        "confusion_matrix.png": 1,
        "per_class_metrics.png": 2,
        "dataset_splits.png": 3,
        "class_distribution.png": 4,
    }
    image_paths = list(figure_root.glob("*.png")) if figure_root.is_dir() else []
    image_paths.sort(key=lambda path: (preferred.get(path.name, 100), path.name.lower()))
    images = [
        ArtifactItem(path=path, kind="image", label=_artifact_label(path), size_bytes=path.stat().st_size)
        for path in image_paths
        if path.is_file()
    ]

    file_paths = [
        path
        for path in task_root.glob("paper/**/*.pdf")
        if "superseded" not in path.parts and path.is_file()
    ]
    publication_manifest = task_root / 'paper/publication_manifest.json'
    if publication_manifest.is_file():
        from .upstream_publication import artifacts, safe_path
        file_paths = [publication_manifest]
        for stage in ('revision_completed', 'translation_completed'):
            try:
                for row in artifacts(task_root,stage):
                    if 'pdf' in row:
                        file_paths.append(safe_path(task_root,row['pdf']))
            except (OSError,ValueError,KeyError):
                # Never offer stale/unverified intermediate PDFs as final outputs.
                continue
    manifest = task_root / "paper" / "figures_generated" / "figure_manifest.json"
    if manifest.is_file():
        file_paths.append(manifest)
    unique_paths = sorted(set(file_paths), key=lambda path: (path.suffix != ".pdf", path.name.lower()))
    files = [
        ArtifactItem(
            path=path,
            kind="pdf" if path.suffix.lower() == ".pdf" else "manifest",
            label=_artifact_label(path),
            size_bytes=path.stat().st_size,
        )
        for path in unique_paths
    ]
    return images, files


def format_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def artifact_placeholder(completed_stage: str) -> tuple[str, str]:
    if completed_stage in {"", "task_created", "dataset_discovered", "dataset_validated"}:
        return "研究尚在准备", "完成实验与绘图阶段后，这里会出现真实图表。"
    if completed_stage in {
        "research_understood",
        "literature_collected",
        "idea_proposed",
        "research_contract_generated",
        "research_contract_approved",
        "experiment_spec_validated",
        "sandbox_prechecked",
    }:
        return "研究方案已在形成", "图表只会在真实实验产生数据后展示。"
    return "成果正在生成", "已完成的真实图表和论文文件会自动汇总到这里。"


def _artifact_label(path: Path) -> str:
    labels = {
        "test_metrics.png": "最终测试指标",
        "confusion_matrix.png": "混淆矩阵",
        "per_class_metrics.png": "逐类别指标",
        "dataset_splits.png": "数据集划分",
        "class_distribution.png": "类别分布",
        "figure_manifest.json": "图表证据清单",
        "final_paper.pdf": "最终论文",
        "template.pdf": "英文论文",
        "translation.pdf": "中文论文",
    }
    return labels.get(path.name, path.stem.replace("_", " ").title())
