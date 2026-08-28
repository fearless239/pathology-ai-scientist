import json
import importlib
import sys
import types


def _install_fake_streamlit(monkeypatch):
    recorded = {"buttons": [], "markdowns": [], "errors": []}

    class FakeColumn:
        def button(self, label, **kwargs):
            recorded["buttons"].append(label)
            results = recorded.get("button_results", {})
            if label in results:
                return results[label]
            return recorded.get("button_result", False)

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    fake = types.ModuleType("streamlit")
    fake.columns = lambda count, **kwargs: [FakeColumn() for _ in range(count)]
    fake.progress = lambda *args, **kwargs: None
    fake.metric = lambda *args, **kwargs: None
    fake.write = lambda *args, **kwargs: None
    fake.markdown = lambda text, **kwargs: recorded["markdowns"].append(text)
    fake.dataframe = lambda *args, **kwargs: None
    fake.divider = lambda *args, **kwargs: None
    fake.subheader = lambda *args, **kwargs: None
    fake.caption = lambda *args, **kwargs: None
    fake.text_input = lambda *args, **kwargs: ""
    fake.button = lambda *args, **kwargs: False
    fake.warning = lambda *args, **kwargs: None
    fake.error = lambda text, **kwargs: recorded["errors"].append(text)
    fake.rerun = lambda *args, **kwargs: None

    class FakeStatus:
        def update(self, *args, **kwargs):
            return None

    fake.status = lambda *args, **kwargs: FakeStatus()
    fake.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return recorded


def test_task_controls_render_control_buttons(project_root, tmp_path, monkeypatch):
    recorded = _install_fake_streamlit(monkeypatch)
    import app
    importlib.reload(app)
    from pathmnist.workflow import WorkflowStore

    root = tmp_path / "state" / "workflow"
    store = WorkflowStore(root)
    store.create(
        "ui-controls",
        {
            "mode": "staged_approval",
            "budget_limit_usd": 2,
            "execution_limit_seconds": 3600,
            "research_direction": "方向",
        },
    )
    app._render_task_controls(project_root, store, "ui-controls")
    assert "推进下一阶段（执行并校验产物）" in recorded["buttons"]
    assert "启动后台执行" in recorded["buttons"]
    assert "暂停" in recorded["buttons"]
    assert "取消" in recorded["buttons"]


def test_sidebar_selection_and_research_preset_update_session_state(monkeypatch):
    _install_fake_streamlit(monkeypatch)
    import app
    importlib.reload(app)

    app._select_research_task("study-002")
    assert app.st.session_state["selected_research_task"] == "study-002"
    assert app.st.session_state["active_view"] == "workspace"

    app._apply_research_preset("研究提示词")
    assert app.st.session_state["research-direction"] == "研究提示词"


def test_topic_proposal_preview_renders_artifacts(project_root, tmp_path, monkeypatch):
    recorded = _install_fake_streamlit(monkeypatch)
    import app
    importlib.reload(app)

    artifacts = tmp_path / "state" / "workflow" / "ui-topic" / "artifacts"
    topic_root = artifacts / "topic_proposed"
    literature_root = artifacts / "literature_collected"
    topic_root.mkdir(parents=True)
    literature_root.mkdir(parents=True)
    (topic_root / "topic.json").write_text(
        json.dumps(
            {
                "title": "拟定题目 A",
                "short_hypothesis": "假设 A",
                "experiments": "实验 A",
                "risk_factors_and_limitations": "风险 A",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (literature_root / "literature.json").write_text(
        json.dumps(
            {"verification_status": "api_verified", "references": [{"title": "p1"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    app._render_topic_proposal(tmp_path, "ui-topic")
    joined = "\n".join(recorded["markdowns"])
    assert "拟定题目 A" in joined
    assert "假设 A" in joined
    assert "api_verified" in joined


def test_missing_llm_key_shows_banner_and_click_fails_gracefully(
    project_root, tmp_path, monkeypatch
):
    recorded = _install_fake_streamlit(monkeypatch)
    monkeypatch.delenv("PARATERA_API_KEY", raising=False)
    import importlib

    import app
    importlib.reload(app)
    from pathmnist.workflow import WorkflowStore

    root = tmp_path / "state" / "workflow"
    store = WorkflowStore(root)
    store.create(
        "ui-no-key",
        {
            "mode": "staged_approval",
            "budget_limit_usd": 2,
            "execution_limit_seconds": 3600,
            "research_direction": "方向",
            "llm_config_path": str(project_root / "configs" / "gate_a_llm.yaml"),
        },
    )
    app._render_task_controls(tmp_path, store, "ui-no-key")
    assert any("PARATERA_API_KEY" in error for error in recorded["errors"])
    recorded["button_result"] = True
    app._render_task_controls(tmp_path, store, "ui-no-key")
    assert any("推进阶段失败" in error for error in recorded["errors"])


def test_auto_advance_button_runs_until_approval_gate(project_root, tmp_path, monkeypatch):
    recorded = _install_fake_streamlit(monkeypatch)
    import importlib

    import app
    importlib.reload(app)
    from pathmnist.workflow import WorkflowStore

    calls = []

    def fake_advance(store, task_id, executor=None, artifact_root=None):
        calls.append(task_id)
        state = store.load(task_id)
        if len(calls) >= 2:
            state.control = "waiting_approval"
            state.stages["topic_approved"]["status"] = "waiting_approval"
        else:
            state.completed_stage = "task_created"
            state.stages["task_created"].update(
                status="completed", outputs={"status": "completed"}
            )
        store.save(state)
        return state

    monkeypatch.setattr(app, "advance", fake_advance)
    monkeypatch.setattr(app, "task_executor", lambda *args, **kwargs: object())
    root = tmp_path / "state" / "workflow"
    store = WorkflowStore(root)
    store.create(
        "ui-auto",
        {
            "mode": "staged_approval",
            "budget_limit_usd": 2,
            "execution_limit_seconds": 3600,
            "research_direction": "方向",
        },
    )
    recorded["button_results"] = {"自动推进到审批点": True}
    app._render_task_controls(tmp_path, store, "ui-auto")
    assert len(calls) == 2


def test_reset_button_retries_interrupted_stage(project_root, tmp_path, monkeypatch):
    recorded = _install_fake_streamlit(monkeypatch)
    import importlib

    import app
    importlib.reload(app)
    from pathmnist.workflow import WorkflowStore

    advanced = []

    def fake_advance(store, task_id, executor=None, artifact_root=None):
        advanced.append(task_id)
        return store.load(task_id)

    monkeypatch.setattr(app, "advance", fake_advance)
    monkeypatch.setattr(app, "task_executor", lambda *args, **kwargs: object())
    root = tmp_path / "state" / "workflow"
    store = WorkflowStore(root)
    store.create(
        "ui-reset",
        {
            "mode": "staged_approval",
            "budget_limit_usd": 2,
            "execution_limit_seconds": 3600,
            "research_direction": "方向",
        },
    )
    state = store.load("ui-reset")
    state.completed_stage = "models_prechecked"
    state.stages["research_understood"].update(status="interrupted", retries=3, error="boom")
    store.save(state)
    recorded["button_results"] = {"重置并重试：方向理解": True}
    app._render_task_controls(tmp_path, store, "ui-reset")
    reset_state = store.load("ui-reset")
    assert reset_state.stages["research_understood"]["retries"] == 0
    assert reset_state.stages["research_understood"]["error"] is None
    assert advanced == ["ui-reset"]
