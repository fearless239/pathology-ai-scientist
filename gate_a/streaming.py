"""Incremental subprocess output with a bounded wait and retained diagnostics."""
import subprocess
import threading


def run_streaming(command, *, timeout, env, on_line):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace", env=env)
    output = {"stdout": [], "stderr": []}

    def consume(name, pipe):
        for line in iter(pipe.readline, ""):
            output[name].append(line)
            on_line(name, line)
        pipe.close()

    threads = [threading.Thread(target=consume, args=(name, getattr(process, name)), daemon=True)
               for name in output]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout)
    except BaseException as error:
        process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=2)
        if isinstance(error, subprocess.TimeoutExpired):
            error.output = "".join(output["stdout"])
            error.stderr = "".join(output["stderr"])
        raise
    for thread in threads:
        thread.join(timeout=2)
    return subprocess.CompletedProcess(command, process.returncode,
                                       "".join(output["stdout"]), "".join(output["stderr"]))
