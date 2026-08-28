from pathlib import Path


def test_research_control_plane_mounts_docker_and_forwards_only_named_key(project_root: Path):
    script = (project_root / "scripts/pathmnist.sh").read_text(encoding="utf-8")
    start = script.index("  v2)")
    end = script.index("  autonomous-init)", start)
    block = script[start:end]
    assert "/var/run/docker.sock" in block
    assert "--env PARATERA_API_KEY" in block
    assert "${PARATERA_API_KEY}" not in block
