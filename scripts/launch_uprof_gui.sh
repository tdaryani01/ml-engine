#!/usr/bin/env bash
# Stable AMD uProf GUI launcher for this project.
#
# Purpose:
# - keep GUI launches as the normal user (not sudo)
# - fail early when BPF/tracefs setup was lost after reboot
# - avoid stale user-owned temp FIFOs confusing repeated GUI runs
# - force AMD's documented GUI service workaround for this host

set -euo pipefail

UPROF_ROOT="${UPROF_ROOT:-/opt/AMDuProf_5.3-521}"
UPROF_GUI="$UPROF_ROOT/bin/AMDuProf"
UPROF_SETUP="$UPROF_ROOT/bin/AMDuProfSetup.sh"
TMP_DIR="/tmp/AMDuProf-${USER}"

if [[ "$(id -u)" == "0" ]]; then
  echo "[ERROR] Do not launch AMDuProf GUI with sudo/root." >&2
  echo "        Run this script as your normal user." >&2
  exit 1
fi

if [[ ! -x "$UPROF_GUI" ]]; then
  echo "[ERROR] AMDuProf GUI not found/executable: $UPROF_GUI" >&2
  exit 1
fi

if [[ ! -r /sys/kernel/tracing/trace ]]; then
  echo "[ERROR] uProf BPF tracing is not readable for this user." >&2
  echo "        This usually resets after reboot. Run:" >&2
  echo "        sudo $UPROF_SETUP" >&2
  exit 1
fi

if [[ -d "$TMP_DIR" && ! -O "$TMP_DIR" ]]; then
  owner="$(stat -c '%U:%G' "$TMP_DIR" 2>/dev/null || echo unknown)"
  echo "[ERROR] $TMP_DIR is not owned by this user ($owner)." >&2
  echo "        Close uProf, then run:" >&2
  echo "        sudo chown -R $USER:$USER $TMP_DIR ~/.AMDuProf" >&2
  exit 1
fi

mkdir -p "$TMP_DIR"

# Remove stale user-owned uProf named pipes from crashed GUI sessions. These are
# recreated by the GUI on launch; leaving dead FIFOs around has caused confusing
# repeated-start behavior.
find "$TMP_DIR" -maxdepth 1 -user "$USER" -type p -name 'uprofport*' -delete 2>/dev/null || true

export AMDUPROF_GUI_WORKAROUND=1
export AMDUPROF_SCALE_FACTOR="${AMDUPROF_SCALE_FACTOR:-1}"

cd "$UPROF_ROOT/bin"
exec "$UPROF_GUI" "$@"
