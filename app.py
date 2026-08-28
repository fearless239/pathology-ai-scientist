from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from pathmnist.workflow import (
    STAGES,
    WorkflowError,
    WorkflowStore,
    task_budget_path,
    task_executor,
    advance,
    approve,
    resume,
    set_control,
    reset_interrupted_stage,
)
from pathmnist.worker import lock_is_stale, release_lock
from path_ai_scientist.core import OrchestrationError, RESEARCH_STAGES, ResearchOrchestrator
from pathmnist.cli import _autonomous_init
from path_ai_scientist.ui.web_runner import log_tail, run_status, start as start_research_run
from path_ai_scientist.ui.research_components import (
    docker_ready as _docker_ready,
    gpu_ready as _gpu_ready,
    readiness_badge as _readiness,
    render_research_downloads as _render_research_downloads,
)
from pathmnist.dashboard_ui import (
    APP_CSS,
    RESEARCH_PRESETS,
    TaskDeletionError,
    delete_research_task,
    list_research_task_summaries,
)


TASK_LABELS = {
    "task_created": "任务创建",
    "dataset_validated": "数据校验",
    "models_prechecked": "模型预检",
    "research_understood": "方向理解",
    "literature_collected": "文献收集",
    "topic_proposed": "选题生成",
    "topic_approved": "选题审批",
    "experiment_planned": "实验计划",
    "budget_approved": "预算审批",
    "baseline_completed": "基线实验",
    "improvements_completed": "改进实验",
    "tuning_completed": "调参实验",
    "ablations_completed": "消融实验",
    "formal_training_approved": "正式训练审批",
    "main_comparison_completed": "主比较",
    "candidate_frozen": "候选冻结",
    "test_evaluated": "一次性测试评估",
    "analysis_completed": "统计分析",
    "paper_approved": "论文审批",
    "english_paper_completed": "英文论文",
    "review_completed": "独立评审",
    "revision_completed": "论文修订",
    "chinese_translation_completed": "中文翻译",
    "archived": "任务归档",
}

CONTROL_LABELS = {
    "running": "运行中",
    "paused": "已暂停",
    "cancelled": "已取消",
    "waiting_approval": "等待审批",
    "completed": "已完成",
}

RESEARCH_STAGE_LABELS = {
    "task_created": "任务创建", "dataset_discovered": "发现数据", "dataset_validated": "校验数据",
    "research_understood": "理解课题", "literature_collected": "检索文献", "idea_proposed": "提出方案",
    "research_contract_generated": "研究合同待确认", "research_contract_approved": "研究合同已批准",
    "experiment_spec_validated": "实验设计", "sandbox_prechecked": "环境预检",
    "initial_implementation_completed": "初始实现", "baseline_tuning_completed": "基线与调参",
    "creative_research_completed": "创新研究", "ablations_completed": "消融实验",
    "candidate_selected": "选择模型", "candidate_frozen": "冻结模型",
    "test_evaluation_approved": "测试审批", "test_evaluated": "最终测试",
    "analysis_completed": "统计分析", "figures_generated": "生成图表", "paper_written": "撰写论文",
    "review_completed": "独立评审", "revision_completed": "修订论文",
    "translation_completed": "中文翻译", "archived": "完成归档",
}

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def workflow_root(project_root: Path) -> Path:
    configured = os.getenv("PATH_AI_SCIENTIST_STATE_ROOT") or os.getenv(
        "PATH_SCIENTIST_STATE_ROOT"
    )
    if not configured:
        return project_root / "state" / "workflow"
    root = Path(configured)
    return root if root.is_absolute() else project_root / root


def _start_worker(project_root: Path, task_id: str) -> None:
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pathmnist.worker",
            task_id,
            "--project-root",
            str(project_root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def render_workflow(project_root: Path) -> None:
    store = WorkflowStore(workflow_root(project_root))
    task_ids = [state.task_id for state in store.list_states()]
    selected_task = st.selectbox("任务", task_ids if task_ids else ["（暂无任务）"])
    if selected_task == "（暂无任务）":
        st.info("尚未创建工作流任务。")
    else:
        _render_task_controls(project_root, store, selected_task)


def _render_task_controls(project_root: Path, store: WorkflowStore, task_id: str) -> None:
    state = store.load(task_id)
    workflow_complete = state.completed_stage == STAGES[-1]
    llm_enabled = bool(state.config.get("llm_config_path"))
    if llm_enabled and not os.getenv("PARATERA_API_KEY", "").strip():
        st.error(
            "该任务启用了 LLM 阶段，但当前容器缺少 PARATERA_API_KEY。"
            "请停止 web，在 WSL 终端 export PARATERA_API_KEY 后重新启动。"
        )
    completed_count = 0 if not state.completed_stage else STAGES.index(state.completed_stage) + 1
    progress = completed_count / len(STAGES)
    st.progress(progress, text=f"阶段进度 {completed_count}/{len(STAGES)}")
    left, middle, right = st.columns(3)
    display_control = "completed" if workflow_complete else state.control
    left.metric("控制状态", CONTROL_LABELS.get(display_control, display_control))
    middle.metric("最近完成阶段", TASK_LABELS.get(state.completed_stage, "尚未开始"))
    right.metric("执行时间", f"{state.execution_seconds:.0f} / {state.config['execution_limit_seconds']} 秒")
    from gate_a.budget import BudgetLedger

    ledger = BudgetLedger(
        task_budget_path(workflow_root(project_root), task_id),
        state.config["budget_limit_usd"],
    )
    snapshot = ledger.snapshot()
    st.write(
        f"预算台账：`${snapshot.spent_usd:.4f} 已花费 + "
        f"${snapshot.reserved_usd:.4f} 已预留 / "
        f"${snapshot.available_usd:.4f} 可用 / ${snapshot.hard_limit_usd:.2f} 硬上限`"
    )
    if state.control == "waiting_approval":
        waiting = next(name for name, item in state.stages.items() if item["status"] == "waiting_approval")
        st.warning(f"等待人工审批：{TASK_LABELS.get(waiting, waiting)}")
        if waiting == "topic_approved":
            _render_topic_proposal(project_root, task_id)
        if st.button("批准该阶段", width="stretch"):
            try:
                approve(
                    store,
                    task_id,
                    waiting,
                    task_executor(project_root, workflow_root(project_root), task_id),
                    workflow_root(project_root) / task_id / "artifacts",
                )
                st.rerun()
            except WorkflowError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"审批阶段失败：{exc}")

    lock_path = workflow_root(Path(__file__).parent) / task_id / "worker.lock"
    if lock_path.exists() and lock_is_stale(lock_path):
        release_lock(lock_path)
    worker_active = lock_path.exists()
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("推进下一阶段（执行并校验产物）", disabled=workflow_complete or state.control != "running", width="stretch"):
        try:
            advance(
                store,
                task_id,
                task_executor(project_root, workflow_root(project_root), task_id),
                workflow_root(project_root) / task_id / "artifacts",
            )
            st.rerun()
        except WorkflowError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"推进阶段失败：{exc}")
    if col2.button("启动后台执行", disabled=workflow_complete or worker_active or state.control != "running", width="stretch"):
        _start_worker(project_root, task_id)
        st.rerun()
    if col3.button("暂停", disabled=workflow_complete or state.control not in {"running", "waiting_approval"}, width="stretch"):
        set_control(store, task_id, "paused")
        st.rerun()
    if col4.button("取消", disabled=workflow_complete or state.control == "cancelled", width="stretch"):
        set_control(store, task_id, "cancelled")
        st.rerun()
    col5, col6 = st.columns(2)
    if col5.button("自动推进到审批点", disabled=workflow_complete or state.control != "running", width="stretch"):
        _auto_advance(project_root, store, task_id)
        st.rerun()
    interrupted = next(
        (name for name, item in state.stages.items() if item["status"] == "interrupted"),
        None,
    )
    if interrupted is not None and col6.button(
        f"重置并重试：{TASK_LABELS.get(interrupted, interrupted)}",
        width="stretch",
    ):
        if worker_active:
            st.error("后台 worker 正在运行；请等它退出或停止 web 后再重置。")
        else:
            reset_interrupted_stage(store, task_id, interrupted)
            try:
                advance(
                    store,
                    task_id,
                    task_executor(project_root, workflow_root(project_root), task_id),
                    workflow_root(project_root) / task_id / "artifacts",
                )
                st.rerun()
            except WorkflowError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"重试失败：{exc}")
    if state.control == "paused" and st.button("从最近完成阶段继续", width="stretch"):
        resume(store, task_id)
        st.rerun()
    rows = []
    for name in STAGES:
        item = state.stages[name]
        rows.append(
            {
                "阶段": TASK_LABELS.get(name, name),
                "状态": item["status"],
                "开始时间": item["started_at"] or "—",
                "完成时间": item["completed_at"] or "—",
                "重试": item["retries"],
                "错误": item["error"] or "—",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    _render_paper_downloads(project_root, task_id)
    st.divider()
    st.subheader("删除任务")
    st.caption("删除会移除该任务的状态、LLM 响应缓存和全部论文产物，无法从网页恢复。")
    confirmation = st.text_input(
        "输入任务 ID 以确认删除",
        key=f"delete-confirm-{task_id}",
        placeholder=task_id,
    )
    if st.button(
        "永久删除任务",
        key=f"delete-task-{task_id}",
        disabled=worker_active or confirmation != task_id,
        width="stretch",
    ):
        try:
            store.delete(task_id)
            st.success(f"任务 {task_id} 已删除。")
            st.rerun()
        except WorkflowError as exc:
            st.error(str(exc))


def _render_paper_downloads(project_root: Path, task_id: str) -> None:
    revision_root = (
        workflow_root(project_root) / task_id / "artifacts" / "revision_completed"
    )
    markdown_path = revision_root / "final_paper.md"
    latex_path = revision_root / "final_paper.tex"
    if not markdown_path.is_file():
        return
    st.subheader("最终论文")
    st.caption("该版本已经过独立评审与修订；Markdown 可直接阅读，LaTeX 可用于正式排版。")
    translation_path = (
        workflow_root(project_root)
        / task_id
        / "artifacts"
        / "chinese_translation_completed"
        / "translation.json"
    )
    left, middle, right = st.columns(3)
    left.download_button(
        "下载最终论文（Markdown）",
        data=markdown_path.read_bytes(),
        file_name=f"{task_id}-final-paper.md",
        mime="text/markdown",
        width="stretch",
    )
    if latex_path.is_file():
        middle.download_button(
            "下载最终论文（LaTeX）",
            data=latex_path.read_bytes(),
            file_name=f"{task_id}-final-paper.tex",
            mime="application/x-tex",
            width="stretch",
        )
    if translation_path.is_file():
        translation = load_json(translation_path)
        chinese_markdown = translation.get("llm_output")
        if chinese_markdown:
            right.download_button(
                "下载中文论文（Markdown）",
                data=chinese_markdown.encode("utf-8"),
                file_name=f"{task_id}-final-paper-zh.md",
                mime="text/markdown",
                width="stretch",
            )


def _auto_advance(project_root: Path, store: WorkflowStore, task_id: str) -> None:
    status = st.status("自动推进中；每个 LLM 阶段可能需要 1–2 分钟，请勿重复点击…")
    advanced = 0
    try:
        while True:
            state = store.load(task_id)
            if state.control != "running":
                status.update(
                    label=(
                        f"已推进 {advanced} 个阶段；当前状态："
                        f"{CONTROL_LABELS.get(state.control, state.control)}"
                    ),
                    state="complete",
                )
                return
            if state.completed_stage == STAGES[-1]:
                status.update(label="任务已完成", state="complete")
                return
            advance(
                store,
                task_id,
                task_executor(project_root, workflow_root(project_root), task_id),
                workflow_root(project_root) / task_id / "artifacts",
            )
            advanced += 1
            status.update(label=f"已推进 {advanced} 个阶段…")
    except WorkflowError as exc:
        status.update(label=f"推进停止（已完成 {advanced} 个阶段）", state="error")
        st.error(str(exc))
    except Exception as exc:
        status.update(label=f"推进停止（已完成 {advanced} 个阶段）", state="error")
        st.error(f"阶段失败：{exc}")


def _render_topic_proposal(project_root: Path, task_id: str) -> None:
    artifacts = workflow_root(project_root) / task_id / "artifacts"
    topic_path = artifacts / "topic_proposed" / "topic.json"
    if not topic_path.is_file():
        return
    topic = json.loads(topic_path.read_text(encoding="utf-8"))
    st.markdown(f"**拟定题目**：{topic.get('title', '—')}")
    st.markdown(f"**研究假设**：{topic.get('short_hypothesis', topic.get('topic', '—'))}")
    st.markdown(f"**实验设计**：{topic.get('experiments', '—')}")
    st.markdown(f"**风险与局限**：{topic.get('risk_factors_and_limitations', '—')}")
    literature_path = artifacts / "literature_collected" / "literature.json"
    if literature_path.is_file():
        literature = json.loads(literature_path.read_text(encoding="utf-8"))
        references = literature.get("references", [])
        st.markdown(
            f"**文献核验**：{literature.get('verification_status', '—')}；"
            f"已核验条目 {len(references)} 条"
        )


def render_task_creator(project_root: Path) -> None:
    flash = st.session_state.pop("task_created_flash", None)
    if flash:
        st.success(flash)
    task_id = st.text_input("任务 ID", value="", placeholder="例如 pathmnist-20260818")
    research_direction = st.text_area(
        "研究方向（必填）",
        placeholder="例如：研究多尺度特征与数据增强对 PathMNIST 验证-测试泛化差距的影响",
    )
    research_goal = st.text_input(
        "研究目标（可选）",
        placeholder="例如：在保持训练预算不变的前提下缩小 Macro-F1 泛化差距",
    )
    mode = st.radio(
        "运行模式",
        ["staged_approval", "autonomous"],
        format_func=lambda value: {
            "staged_approval": "分阶段人工审批",
            "autonomous": "全自动运行",
        }[value],
        horizontal=True,
    )
    budget = st.number_input("预算硬上限（USD）", 0.1, 50.0, 2.0, step=0.1)
    execution_hours = st.number_input("执行时限（小时）", 0.1, 6.0, 6.0, step=0.1)
    enable_real_training = st.toggle(
        "高级：重新运行仓库内已实现的训练变体",
        value=False,
        help=(
            "关闭时复用已冻结的 PathMNIST 实验，适合端到端演示。开启时会占用 GPU/磁盘，"
            "但只会训练仓库已实现的 augmentation、optimization、multiscale 和 combined；"
            "不会自动实现研究方向中提出的全新网络或路由算法。"
        ),
    )
    enable_llm = st.toggle("启用论文链路 LLM", value=False)
    llm_config_path = (
        str(project_root / "configs" / "gate_a_llm.yaml") if enable_llm else ""
    )
    if st.button("创建并启动任务", width="stretch"):
        if not task_id.strip():
            st.error("请先输入任务 ID。")
            return
        if not research_direction.strip():
            st.error("请先输入研究方向（FR-001）。")
            return
        store = WorkflowStore(workflow_root(project_root))
        try:
            store.create(
                task_id.strip(),
                {
                    "mode": mode,
                    "budget_limit_usd": budget,
                    "execution_limit_seconds": int(execution_hours * 3600),
                    "enable_real_training": enable_real_training,
                    "llm_config_path": llm_config_path,
                    "research_direction": research_direction.strip(),
                    "research_goal": research_goal.strip(),
                },
            )
            _start_worker(project_root, task_id.strip())
            st.session_state["task_created_flash"] = (
                f"任务 {task_id.strip()} 已创建并在后台启动。"
                "请切换到「任务控制」标签页查看进度。"
            )
            st.rerun()
        except WorkflowError as exc:
            st.error(str(exc))
        except OSError as exc:
            st.error(f"创建任务失败（无法写入状态文件）：{exc}")


def _set_app_view(view: str) -> None:
    st.session_state["active_view"] = view


def _select_research_task(task_id: str) -> None:
    st.session_state["selected_research_task"] = task_id
    st.session_state["active_view"] = "workspace"


def _apply_research_preset(prompt: str) -> None:
    st.session_state["research-direction"] = prompt


def _render_sidebar(project_root: Path) -> None:
    state_root = workflow_root(project_root)
    summaries = list_research_task_summaries(state_root)
    selected = st.session_state.get("selected_research_task")
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="sidebar-mark">P</div>
              <div><strong>Path-AI Scientist</strong><span>病理 AI 科研工作台</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "＋ 新建课题",
            key="sidebar-create",
            type="primary",
            on_click=_set_app_view,
            args=("create",),
            width="stretch",
        )
        st.button(
            "◉ 实验中心",
            key="sidebar-workspace",
            on_click=_set_app_view,
            args=("workspace",),
            width="stretch",
        )
        st.markdown('<div class="sidebar-section">最近课题</div>', unsafe_allow_html=True)
        if not summaries:
            st.caption("还没有课题，从一次清晰的研究提问开始。")
        for item in summaries:
            background = run_status(state_root / item.task_id)
            run_state = background.get("state", "not_started")
            icon = {
                "running": "●",
                "completed": "✓",
                "failed": "!",
                "interrupted": "↻",
            }.get(run_state, "○")
            button_type = "primary" if item.task_id == selected else "secondary"
            st.button(
                f"{icon}  {item.task_id}",
                key=f"sidebar-task-{item.task_id}",
                type=button_type,
                on_click=_select_research_task,
                args=(item.task_id,),
                width="stretch",
            )
            stage = RESEARCH_STAGE_LABELS.get(item.completed_stage, item.completed_stage or "尚未开始")
            direction = html.escape(item.research_direction or "未填写研究方向")
            st.markdown(
                f'<div class="sidebar-task-meta" title="{direction}">{stage} · {direction}</div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    project_root = Path(__file__).parent
    st.set_page_config(
        page_title="Path-AI Scientist",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    summaries = list_research_task_summaries(workflow_root(project_root))
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "workspace" if summaries else "create"
    if summaries and st.session_state.get("selected_research_task") not in {
        item.task_id for item in summaries
    }:
        st.session_state["selected_research_task"] = summaries[0].task_id
    _render_sidebar(project_root)
    if st.session_state.get("active_view") == "create":
        render_research_task_creator(project_root)
    else:
        render_research_workflow(project_root)


def _research_task_ids(project_root: Path) -> list[str]:
    return [item.task_id for item in list_research_task_summaries(workflow_root(project_root))]


def render_research_workflow(project_root: Path) -> None:
    tasks = _research_task_ids(project_root)
    if not tasks:
        st.markdown(
            """
            <div class="empty-state">
              <div>
                <div class="page-eyebrow">Research workspace</div>
                <h2>还没有研究课题</h2>
                <p>从一个可验证的病理 AI 问题开始，系统会把实验、证据、图表与论文组织在同一个工作区。</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "＋ 创建第一个课题",
            type="primary",
            on_click=_set_app_view,
            args=("create",),
            width="stretch",
        )
        return
    preferred = st.session_state.pop("new_research_task", None)
    if preferred in tasks:
        st.session_state["selected_research_task"] = preferred
    task_id = st.session_state.get("selected_research_task")
    if task_id not in tasks:
        task_id = tasks[0]
        st.session_state["selected_research_task"] = task_id
    orchestrator = ResearchOrchestrator(project_root, workflow_root(project_root), task_id)
    try:
        status = orchestrator.status()
    except OrchestrationError as exc:
        st.error(str(exc))
        return
    completed = status["completed_stage"]
    count = RESEARCH_STAGES.index(completed) + 1 if completed in RESEARCH_STAGES else 0
    background = run_status(orchestrator.task_root)
    run_state = background.get("state", "not_started")
    if completed == "archived":
        run_state = "completed"
        background["message"] = "实验、绘图、双语论文和归档均已完成"
    state_names = {
        "not_started": "尚未启动",
        "running": "后台运行中",
        "completed": "全部完成",
        "failed": "需要处理",
        "interrupted": "已中断",
        "unknown": "状态未知",
    }
    task_value = load_json(orchestrator.task_root / "task.json")
    direction = html.escape(str(task_value.get("research_direction", "未填写研究方向")))
    state_class = ""
    if run_state in {"failed"}:
        state_class = " danger"
    elif run_state in {"not_started", "interrupted", "unknown"}:
        state_class = " warning"
    flash = st.session_state.pop("task_created_flash", None)
    if flash:
        st.success(flash)
    st.markdown(
        f"""
        <div class="task-header">
          <div class="task-header-row">
            <div>
              <div class="page-eyebrow">Active research</div>
              <h1>{html.escape(task_id)}</h1>
              <p>{direction}</p>
            </div>
            <span class="status-pill{state_class}">{state_names.get(run_state, run_state)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(count / len(RESEARCH_STAGES), text=f"研究进度 {count}/{len(RESEARCH_STAGES)}")
    st.markdown("### 研究状态")
    left, middle, right, fourth = st.columns(4)
    left.metric("运行状态", state_names.get(run_state, run_state))
    middle.metric("已完成", RESEARCH_STAGE_LABELS.get(completed, completed))
    right.metric("下一步", RESEARCH_STAGE_LABELS.get(status["next_stage"], status["next_stage"] or "—"))
    fourth.metric("证据校验", "通过" if status["valid"] else "异常")
    agent_progress = background.get("progress", {})
    if agent_progress.get("stage"):
        st.caption(
            f"最近实验阶段：{agent_progress['stage']} · "
            f"记录数：{agent_progress.get('node_count', 0)} · "
            f"有效运行：{agent_progress.get('good_nodes', 0)} · "
            f"更新时间：{agent_progress.get('updated_at', '未知')}。"
            "这是执行进度，不代表该科研阶段已通过验收。"
        )
    integrity_names = {"passed": "通过", "failed": "失败", "not_evaluated": "待验证"}
    publication_names = {
        "research_paper": "Research Paper",
        "failure_diagnosis": "Failure Diagnosis",
        "not_applicable": "待确定",
    }
    st.markdown("### 科研完整性")
    data_col, test_col, metric_col, publication_col = st.columns(4)
    data_col.metric("真实数据使用", integrity_names.get(status["data_integrity"], status["data_integrity"]))
    test_col.metric("Sealed test 消费", integrity_names.get(status["sealed_test_integrity"], status["sealed_test_integrity"]))
    metric_col.metric("可信指标", integrity_names.get(status["metric_integrity"], status["metric_integrity"]))
    publication_col.metric("论文输出类型", publication_names.get(status["publication_mode"], status["publication_mode"]))
    contract_col, comparison_col, repeat_col, stats_col = st.columns(4)
    contract_status = status.get("research_contract_integrity", "not_evaluated")
    contract_col.metric("研究合同", integrity_names.get(contract_status, contract_status))
    comparison_status = status.get("comparison_integrity", "not_evaluated")
    comparison_col.metric("主比较", integrity_names.get(comparison_status, comparison_status))
    repeat_status = status.get("repeat_integrity", "not_evaluated")
    repeat_col.metric("重复实验", integrity_names.get(repeat_status, repeat_status))
    statistics_status = status.get("statistical_integrity", "not_evaluated")
    stats_col.metric("统计检验", integrity_names.get(statistics_status, statistics_status))
    message = html.escape(str(background.get("message", "")))
    st.markdown(f'<div class="stage">{message}</div>', unsafe_allow_html=True)
    if status["errors"]:
        st.error("；".join(status["errors"]))
    api_ready = bool(os.getenv("PARATERA_API_KEY", "").strip())
    gpu_ready = _gpu_ready()
    docker_ready = _docker_ready()
    st.markdown("### 运行与授权")
    contract_waiting = completed == "research_contract_generated"
    contract_reviewable = completed in {"research_contract_generated", "experiment_spec_validated", "sandbox_prechecked"}
    if contract_reviewable:
        contract_path = orchestrator.task_root / "research/research_contract.json"
        if contract_path.is_file():
            contract = load_json(contract_path)
            st.markdown("### 研究执行合同确认")
            left_contract, right_contract = st.columns(2)
            with left_contract:
                st.markdown("#### 原始研究方向")
                st.write(task_value.get("research_direction", ""))
            with right_contract:
                st.markdown("#### 结构化实验要求")
                st.write(f"**基线：** {contract.get('baseline', {}).get('name', '—')}")
                from gate_a.model_contract import input_sizes
                try:
                    sizes = input_sizes(contract)
                    st.write("**模型输入尺寸：** " + (" / ".join(f"{h}×{w}" for h, w in sizes) if sizes else "未约定固定尺寸"))
                except ValueError as error:
                    st.error(str(error))
                st.write("**干预：** " + "；".join(str(item.get("name", "")) for item in contract.get("interventions", [])))
                primary = contract.get("metrics", {}).get("primary", {})
                st.write(f"**主指标：** {primary.get('name', '—')}（{primary.get('scope', '—')}）")
                repeat = contract.get("repeat_plan", {})
                st.write(f"**重复实验：** {repeat.get('count', '—')} 次，训练 seeds={repeat.get('seeds', [])}")
                criterion = (contract.get("success_criteria") or [{}])[0]
                st.write(f"**成功阈值：** {criterion.get('minimum_delta') if criterion.get('minimum_delta') is not None else '未预设，仅报告结果'}")
                if not contract.get("capability", {}).get("supported", False):
                    st.error("；".join(contract.get("capability", {}).get("reasons", [])))
            feedback = st.text_area(
                "合同修订意见（可选）",
                key=f"contract-feedback-{task_id}",
                placeholder="例如：重复次数应为 5，主指标应为类别对平均 F1……",
            )
            approve_col, revise_col = st.columns(2)
            if approve_col.button(
                "批准研究合同",
                type="primary",
                disabled=not contract_waiting or not contract.get("capability", {}).get("supported", False),
                width="stretch",
            ):
                try:
                    orchestrator.approve_research_contract()
                    st.success("研究合同已批准。现在可以启动完整付费实验。")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            revision_paid_authorized = bool(st.session_state.get(f"paid-{task_id}")) and api_ready
            if revise_col.button("根据意见重新生成", disabled=not feedback.strip() or not revision_paid_authorized, width="stretch"):
                try:
                    orchestrator.revise_research_contract(feedback.strip())
                    st.success("研究合同已根据意见重新生成，请再次核对。")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with st.container(border=True, key="run-control-card"):
        st.caption("启动前确认运行环境和高成本操作；所有授权仍受项目预算与科研完整性约束。")
        c1, c2, c3 = st.columns(3)
        c1.markdown(_readiness("GPU", gpu_ready), unsafe_allow_html=True)
        c2.markdown(_readiness("AI API", api_ready), unsafe_allow_html=True)
        c3.markdown(_readiness("Docker / PDF", docker_ready), unsafe_allow_html=True)
        paid = st.checkbox("我确认允许调用付费 AI API（仍受项目预算上限保护）", key=f"paid-{task_id}")
        sealed = st.checkbox("我确认允许系统自动执行唯一一次 sealed test", key=f"test-{task_id}")
        compute = st.checkbox("我确认允许使用本机 GPU 和 Docker", key=f"gpu-{task_id}")
    all_confirmed = paid and sealed and compute and api_ready and gpu_ready and docker_ready and not contract_waiting
    if run_state != "completed":
        label = "继续完整实验" if run_state in {"failed", "interrupted"} else "启动完整实验"
        with st.container(border=True, key="task-action-bar"):
            action_copy, action_button = st.columns([3, 1])
            if run_state == "running":
                action_copy.caption("后台研究正在运行；关闭页面不会中断任务。")
            elif not all_confirmed:
                action_copy.caption("三个环境均可用并完成三项授权后，主操作才会开放。")
            else:
                action_copy.caption("准备就绪。运行会从最近一个通过校验的阶段开始。")
            if action_button.button(
                label,
                type="primary",
                disabled=run_state == "running" or not all_confirmed,
                width="stretch",
            ):
                try:
                    start_research_run(project_root, workflow_root(project_root), task_id)
                    st.success("已交给后台运行。现在可以关闭页面，稍后回来查看结果。")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    output = log_tail(orchestrator.task_root)
    if output:
        with st.expander("实时运行日志", expanded=run_state in {"failed", "interrupted"}):
            st.code(output, language="text")
    _render_research_downloads(orchestrator.task_root, completed)
    with st.expander("删除当前课题"):
        st.caption("删除会移除该课题的状态、运行日志和全部生成产物，且无法从网页恢复。")
        confirmation = st.text_input(
            "输入课题 ID 以确认删除",
            key=f"delete-research-confirm-{task_id}",
            placeholder=task_id,
        )
        if st.button(
            "永久删除课题",
            key=f"delete-research-task-{task_id}",
            disabled=run_state == "running",
        ):
            if confirmation != task_id:
                st.error("课题 ID 不匹配，未执行删除。")
            else:
                try:
                    delete_research_task(workflow_root(project_root), task_id)
                except TaskDeletionError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.pop("selected_research_task", None)
                    st.session_state["active_view"] = "workspace"
                    st.success(f"课题 {task_id} 已删除。")
                    st.rerun()
    with st.expander("技术状态（故障排查用）"):
        st.json(status)
    if run_state == "running":
        st.caption("页面每 5 秒自动更新；关闭网页不会停止后台实验。")
        time.sleep(5)
        st.rerun()


def render_research_task_creator(project_root: Path) -> None:
    st.markdown(
        """
        <div class="page-intro">
          <div class="page-eyebrow">New research</div>
          <h1>今天想研究什么？</h1>
          <p>先描述一个可以被实验验证的问题。你可以从科研场景开始，也可以直接写下自己的病理 AI 研究方向。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 从一个科研场景开始")
    preset_columns = st.columns(len(RESEARCH_PRESETS))
    for column, preset in zip(preset_columns, RESEARCH_PRESETS):
        with column:
            with st.container(border=True, key=f"preset-{preset.slug}"):
                st.markdown(f"### {preset.title}")
                st.markdown(preset.description)
                st.button(
                    "使用这个方向 →",
                    key=f"use-preset-{preset.slug}",
                    on_click=_apply_research_preset,
                    args=(preset.prompt,),
                    width="stretch",
                )
    st.markdown("### 描述你的课题")
    st.caption("所有新课题使用完整研究模式：API 硬上限 $8，接近 AI-Scientist-v2 原始搜索额度，并由研究合同决定是否完成。")
    with st.form("create-research-task"):
        direction = st.text_area(
            "研究方向",
            placeholder="例如：研究更稳健的病理图像多分类方法……",
            key="research-direction",
            height=150,
        )
        task_id = st.text_input(
            "课题 ID", placeholder="pathology-study-001", key="research-task-id"
        )
        with st.expander("数据与复现设置"):
            dataset = st.text_input(
                "数据集路径",
                value=str(project_root / "pathmnist_64.npz"),
                help="支持 NPZ、ImageFolder 或带 split/group 字段的 manifest 数据集。",
            )
            seed = st.number_input(
                "自动拆分随机种子",
                min_value=0,
                value=7,
                help="仅当数据没有完整 train/validation/test 时用于可复现拆分；若数据已有官方 split（例如 PathMNIST NPZ），此值会被忽略。",
            )
        submitted = st.form_submit_button("创建课题", type="primary", width="stretch")
    if submitted:
        if not task_id.strip() or not direction.strip():
            st.error("任务 ID 和研究方向不能为空。")
            return
        try:
            result = _autonomous_init(
                argparse.Namespace(
                    state_root=workflow_root(project_root), task_id=task_id.strip(),
                    dataset_path=Path(dataset), direction=direction.strip(), seed=int(seed), resume=False,
                )
            )
            st.session_state["selected_research_task"] = result["task_id"]
            st.session_state["active_view"] = "workspace"
            st.session_state["task_created_flash"] = (
                f"已创建 {result['task_id']}，test 未进入研究视图。"
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
