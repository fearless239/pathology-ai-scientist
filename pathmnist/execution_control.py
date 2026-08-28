"""Cross-process, automatically released task ownership."""
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from functools import wraps

_owners = threading.local()


def task_operation(function):
    @wraps(function)
    def guarded(project_root, state_root, task_id, *args, **kwargs):
        with task_lock(Path(state_root) / task_id, reentrant=True):
            return function(project_root, state_root, task_id, *args, **kwargs)
    return guarded


@contextmanager
def task_lock(root: Path, *, reentrant: bool = False):
    root = root.resolve()
    held = getattr(_owners, 'roots', set())
    if root in held:
        if not reentrant:
            raise RuntimeError("Task already has an active runner; duplicate start refused")
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Task already has an active runner; duplicate start refused") from error
        try:
            _owners.roots = held | {root}
            yield
        finally:
            _owners.roots = held
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
