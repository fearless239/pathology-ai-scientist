from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path

import streamlit as st

from pathmnist.dashboard_ui import (
    artifact_placeholder,
    discover_artifacts,
    format_bytes,
)


def gpu_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


def docker_ready() -> bool:
    if shutil.which("docker") is None or not Path("/var/run/docker.sock").exists():
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def readiness_badge(name: str, ready: bool) -> str:
    css = "ready" if ready else "not-ready"
    mark = "✓ 可用" if ready else "✕ 未连接"
    return f'<div class="{css}"><b>{name}</b><br>{mark}</div>'


def render_research_downloads(task_root: Path, completed_stage: str = "") -> None:
    images, files = discover_artifacts(task_root)
    st.markdown("### 研究成果")
    st.caption("这里只展示研究流程真实生成的图表与文件，不使用装饰性数据替代实验结果。")
    if images:
        columns = st.columns(3)
        for index, item in enumerate(images):
            columns[index % 3].image(
                str(item.path),
                caption=f"{item.label} · {format_bytes(item.size_bytes)}",
                width="stretch",
            )
    else:
        title, copy = artifact_placeholder(completed_stage)
        st.markdown(
            f'<div class="artifact-placeholder"><strong>{title}</strong><span>{copy}</span></div>',
            unsafe_allow_html=True,
        )
    if not files:
        return
    st.markdown("#### 可下载文件")
    columns = st.columns(3)
    for index, item in enumerate(files):
        column = columns[index % 3]
        kind = "PDF" if item.kind == "pdf" else "JSON"
        column.markdown(
            f"""
            <div class="artifact-file">
              <strong>{html.escape(item.label)}</strong>
              <span>{kind} · {format_bytes(item.size_bytes)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        column.download_button(
            f"下载 {item.path.name}",
            item.path.read_bytes(),
            file_name=item.path.name,
            mime="application/pdf" if item.kind == "pdf" else "application/json",
            key=f"download-{item.path}",
            width="stretch",
        )


__all__ = ["docker_ready", "gpu_ready", "readiness_badge", "render_research_downloads"]
