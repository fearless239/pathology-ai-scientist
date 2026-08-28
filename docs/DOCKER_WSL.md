# Docker on Windows with WSL 2

Path-AI Scientist supports a Docker Engine installed directly inside an Ubuntu WSL 2 distribution.
Docker Desktop is optional and is not required for this setup.

## Confirm the existing engine

Open Ubuntu, or enter it from PowerShell:

```powershell
wsl -d Ubuntu-24.04
```

Run these commands inside Ubuntu:

```bash
docker version
docker compose version
docker info
```

`docker version` must show both a client and a server. If `docker info` reports a permission error,
either configure the current user for the Docker group or use the installation's documented `sudo`
workflow. Never put a sudo password in `.env`, a script, Git, or a chat message.

## Build from this Windows checkout

When the repository is on a Windows-mounted drive, BuildKit can fail while reading Windows xattrs or ACLs from
cache directories, even if `.dockerignore` excludes them. The supplied helper creates a minimal,
temporary Linux-side build context and removes it after the build:

```bash
cd <WSL-PATH-TO-REPOSITORY>
bash scripts/build-demo-wsl.sh . path-ai-scientist-demo:local
```

For a different checkout, replace the `cd` path. A repository cloned directly into the WSL Linux
filesystem can also use the ordinary Compose path:

```bash
docker compose up --build
```

## Verify the image

The verification script checks all release-critical Demo guarantees:

```bash
bash scripts/verify-docker.sh path-ai-scientist-demo:local
```

Expected output includes:

```text
NON_ROOT_UID=10001
"passed": true
READ_ONLY_CHECK=passed
SECRET_CHECK=passed
STREAMLIT_HEALTH=ok
```

The script verifies that the deterministic Demo works without networking, `/app` is read-only,
common provider keys are absent, and Streamlit answers its health endpoint. Its temporary container
is removed automatically.

## Run the UI

```bash
docker run --rm \
  --name path-ai-scientist-demo \
  --read-only \
  --tmpfs /tmp:uid=10001,gid=10001 \
  --tmpfs /app/.demo:uid=10001,gid=10001 \
  -p 8501:8501 \
  path-ai-scientist-demo:local
```

Open <http://127.0.0.1:8501>. Stop it with `Ctrl+C`.

## PowerShell usage

In this setup, `docker` belongs to Ubuntu and is not necessarily available directly in PowerShell.
Either enter Ubuntu first, or invoke a single command through WSL:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd <WSL-PATH-TO-REPOSITORY> && bash scripts/verify-docker.sh path-ai-scientist-demo:local"
```

Do not install Docker Desktop merely because PowerShell says that `docker` is not recognized. First
check whether Docker already works inside the WSL distribution.

## Troubleshooting

### `docker: command not found` in PowerShell

This does not prove Docker is absent. Run `wsl --list --verbose`, enter the intended distribution, and
try `docker version` there.

### `failed to xattr ... permission denied`

This is a Windows-mounted-directory build-context problem. Use `scripts/build-demo-wsl.sh`, or clone
the repository into the WSL Linux filesystem.

### Cannot connect to the Docker daemon

Check the daemon inside Ubuntu:

```bash
sudo service docker status
sudo service docker start
```

These commands may prompt locally for your sudo password. Do not disclose the password elsewhere.

### Port 8501 is already allocated

Find the existing container with `docker ps`, stop it if appropriate, or map another host port such as
`-p 8502:8501` and open <http://127.0.0.1:8502>.
