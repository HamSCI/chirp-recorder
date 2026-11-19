#!/bin/bash
# install.sh — first-run bootstrap for chirp-recorder (Pattern A editable install)
#
# Usage: sudo ./scripts/install.sh [--pull] [--yes]
#
# What it does:
#   1. Creates service user chirprec:chirprec
#   2. Clones/links repo to /opt/git/sigmond/chirp-recorder
#   3. Creates venv at /opt/git/sigmond/chirp-recorder/venv with editable install
#   4. Creates /etc/chirp-recorder (per-instance configs only; renders none)
#   5. Installs systemd unit template
#
# Idempotent: safe to re-run.

set -euo pipefail

SERVICE_USER="chirprec"
SERVICE_GROUP="chirprec"
REPO_SOURCE="/opt/git/sigmond/chirp-recorder"
VENV_DIR="/opt/git/sigmond/chirp-recorder/venv"
CONFIG_DIR="/etc/chirp-recorder"
SPOOL_DIR="/var/lib/chirp-recorder"
LOG_DIR="/var/log/chirp-recorder"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ui_info()  { echo "[INFO]  $*"; }
ui_warn()  { echo "[WARN]  $*" >&2; }
ui_error() { echo "[ERROR] $*" >&2; }

DO_PULL=false
AUTO_YES=false
for arg in "$@"; do
    case "$arg" in
        --pull) DO_PULL=true ;;
        --yes)  AUTO_YES=true ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    ui_error "Must run as root (sudo)"
    exit 1
fi

# --- Phase 0.5: ensure uv is on PATH (canonical sigmond-suite installer) ---
# Delegates to sigmond's shared helper if present; inline fallback for
# the bootstrap case.  Keep the fallback in sync with
# sigmond/scripts/install/ensure_uv.sh.
_ENSURE_UV_SH="/opt/git/sigmond/sigmond/scripts/install/ensure_uv.sh"
if [[ -r "$_ENSURE_UV_SH" ]]; then
    # shellcheck source=/dev/null
    source "$_ENSURE_UV_SH"
else
    # Place uv-managed Pythons in a world-readable shared location so
    # service users can execute the venv interpreter.  Under sudo the
    # default ~/.local/share/uv/python resolves to /root/.local/..., and
    # /root is mode 0700, so the venv's python symlink dangles for them.
    # Same value as sigmond's helper; keep in sync with it.
    export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/uv/python}"

    _ensure_uv() {
        # uv refuses to install into a missing parent, and the directory
        # must be world-readable (0755) for service users to traverse it.
        if [[ ! -d "$UV_PYTHON_INSTALL_DIR" ]]; then
            if [[ $EUID -eq 0 ]]; then
                install -d -m 0755 "$UV_PYTHON_INSTALL_DIR" 2>/dev/null || true
            else
                sudo install -d -m 0755 "$UV_PYTHON_INSTALL_DIR" 2>/dev/null || true
            fi
        fi
        if command -v uv >/dev/null 2>&1; then
            printf '[INFO]  uv %s at %s\n' "$(uv --version 2>/dev/null | awk '{print $2}')" "$(command -v uv)"
            return 0
        fi
        printf '[INFO]  uv not found -- installing system-wide to /usr/local/bin\n'
        command -v curl >/dev/null || { printf '[ERROR] curl not found (apt install curl)\n' >&2; return 1; }
        if ! curl -LsSf https://astral.sh/uv/install.sh | env XDG_BIN_HOME=/usr/local/bin UV_NO_MODIFY_PATH=1 sh; then
            printf '[ERROR] uv installer failed\n' >&2
            return 1
        fi
        command -v uv >/dev/null || { printf '[ERROR] uv installer ran but uv is still not on PATH\n' >&2; return 1; }
        printf '[INFO]  uv %s installed\n' "$(uv --version 2>/dev/null | awk '{print $2}')"
    }
fi
_ensure_uv || { ui_error "_ensure_uv failed"; exit 1; }

# --- Phase 0.6: ensure ka9q-python sibling repo is on disk ---
# pyproject.toml's [tool.uv.sources] declares ka9q-python as a path-based
# editable dep at ../ka9q-python.  uv sync needs the directory to exist
# at /opt/git/sigmond/ka9q-python or it fails.
if [[ ! -f /opt/git/sigmond/ka9q-python/pyproject.toml ]]; then
    ui_info "ka9q-python sibling repo not at /opt/git/sigmond/ka9q-python -- cloning"
    mkdir -p /opt/git/sigmond
    git clone https://github.com/HamSCI/ka9q-python /opt/git/sigmond/ka9q-python \
        || { ui_error "Failed to clone ka9q-python"; exit 1; }
fi

# hamsci-dsp is the suite-shared DSP/geometry sibling (also path-based
# editable in [tool.uv.sources]); uv sync needs it on disk too.  It's
# local-only (not on PyPI), so a host without it must clone here or the
# sync fails to resolve ../hamsci-dsp.
if [[ ! -f /opt/git/sigmond/hamsci-dsp/pyproject.toml ]]; then
    ui_info "hamsci-dsp sibling repo not at /opt/git/sigmond/hamsci-dsp -- cloning"
    mkdir -p /opt/git/sigmond
    git clone https://github.com/HamSCI/hamsci-dsp /opt/git/sigmond/hamsci-dsp \
        || { ui_error "Failed to clone hamsci-dsp"; exit 1; }
fi

# hs-uploader supplies the sink writer (hs_uploader.sink.Writer); also a
# path-based editable sibling, so uv sync needs it on disk.
if [[ ! -f /opt/git/sigmond/hs-uploader/pyproject.toml ]]; then
    ui_info "hs-uploader sibling repo not at /opt/git/sigmond/hs-uploader -- cloning"
    mkdir -p /opt/git/sigmond
    git clone https://github.com/HamSCI/hs-uploader /opt/git/sigmond/hs-uploader \
        || { ui_error "Failed to clone hs-uploader"; exit 1; }
fi

# --- Phase 1: service user ---
if ! id -u "$SERVICE_USER" &>/dev/null; then
    ui_info "Creating service user $SERVICE_USER"
    useradd --system --shell /usr/sbin/nologin \
            --home-dir /nonexistent --no-create-home \
            "$SERVICE_USER"
fi

# The sink (/var/lib/sigmond/sink.db, root:sigmond 0664, dir 2775) takes the
# codar.spots rows and the sigmondb chirp rows.  Without the sigmond group
# every insert fails "attempt to write a readonly database" (KFS-MAIN held no
# codar rows at all, 2026-11-12, the finding that put this step here).  Same step as hfdl-recorder and psk-recorder;
# the group comes from sigmond/install.sh, so re-run this afterwards if absent.
if getent group sigmond &>/dev/null; then
    if ! id -nG "$SERVICE_USER" 2>/dev/null | tr ' ' '\n' | grep -qx sigmond; then
        usermod -a -G sigmond "$SERVICE_USER"
        ui_info "Added $SERVICE_USER to sigmond group"
    fi
else
    ui_info "sigmond group not present yet -- re-run after sigmond install"
fi

# --- Phase 2: repo + venv ---
if [[ ! -d "$REPO_SOURCE" ]] && [[ ! -L "$REPO_SOURCE" ]]; then
    ui_info "Linking $REPO_ROOT -> $REPO_SOURCE"
    mkdir -p "$(dirname "$REPO_SOURCE")"
    ln -sfn "$REPO_ROOT" "$REPO_SOURCE"
fi

# Pattern A traversability check
if ! sudo -u "$SERVICE_USER" test -r "$REPO_SOURCE/src/chirp_recorder/__init__.py"; then
    ui_error "Service user $SERVICE_USER cannot read $REPO_SOURCE/src/chirp_recorder/__init__.py"
    ui_error "Fix: ensure the repo is at /opt/git/sigmond/chirp-recorder (not under a mode-700 home),"
    ui_error "  or chmod g+rx the path and add $SERVICE_USER to the owner's group"
    exit 1
fi

if $DO_PULL; then
    ui_info "Pulling latest from origin"
    git -C "$REPO_SOURCE" pull --ff-only
fi

if [[ ! -d "$VENV_DIR" ]]; then
    ui_info "Creating venv at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    # --seed populates pip/setuptools/wheel for compatibility with tooling
    # that shells out to pip; harmless overhead otherwise.
    uv venv "$VENV_DIR" --python 3.11 --seed --quiet
fi

# uv sync reads pyproject.toml + uv.lock, resolves [tool.uv.sources]
# (ka9q-python editable from ../ka9q-python), installs chirp-recorder
# itself editable, and pins exactly what's in uv.lock.  --no-dev skips
# dev extras (pytest etc.); --frozen requires uv.lock to be current
# (regenerate locally with `uv lock` if siblings or deps have shifted).
ui_info "Syncing chirp-recorder + ka9q-python (editable) into $VENV_DIR"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" \
    uv sync --project "$REPO_SOURCE" --frozen --no-dev --quiet

# sigmond is the host-wide orchestrator.  chirp-recorder's sink writer falls
# back to sigmond.hamsci_sink.Writer when hs_uploader does not import.  The
# pyproject does not declare sigmond, so uv sync does not install it; this
# step installs it editable when the sibling exists.  uv pip install needs
# --python (UV_PROJECT_ENVIRONMENT only applies to project commands like
# uv sync).
if [[ -d /opt/git/sigmond/sigmond ]]; then
    ui_info "Installing sigmond (editable) into venv"
    uv pip install --quiet --python "$VENV_DIR/bin/python3" -e /opt/git/sigmond/sigmond
else
    ui_info "sigmond repo not found at /opt/git/sigmond/sigmond -- the sink writer"
    ui_info "  uses hs_uploader alone."
fi

# Post-install verify
if ! sudo -u "$SERVICE_USER" "$VENV_DIR/bin/python3" -c 'import chirp_recorder' 2>/dev/null; then
    ui_error "Post-install verify failed: $SERVICE_USER cannot import chirp_recorder"
    exit 1
fi
ui_info "Post-install verify OK"

# --- Phase 3: config ---
# Per-instance config only: /etc/chirp-recorder/<reporter-key>.toml.  There is
# no shared config, and nothing renders one.
mkdir -p "$CONFIG_DIR"

# --- Phase 4: directories ---
for dir in "$SPOOL_DIR" "$LOG_DIR"; do
    mkdir -p "$dir"
    chown "$SERVICE_USER:$SERVICE_GROUP" "$dir"
done

# --- Phase 5: systemd ---
ui_info "Installing systemd unit template"
install -o root -g root -m 644 \
    "$REPO_SOURCE/systemd/chirp-recorder@.service" \
    /etc/systemd/system/chirp-recorder@.service
systemctl daemon-reload

ui_info "Install complete. Write one config per receiver, then enable it with:"
ui_info "  sudo systemctl enable --now chirp-recorder@<reporter-key>"
ui_info "  (config: $CONFIG_DIR/<reporter-key>.toml)"
