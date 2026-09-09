#!/usr/bin/env bash
# Guarantee the venv is on PATH no matter how the container is invoked.
# `bash -lc` (a LOGIN shell) re-sources /etc/profile and discards the image's ENV PATH,
# which silently made every swoosh-* command "not found" inside the container.
set -e
export PATH="/opt/venv/bin:/root/.local/bin:${PATH}"
export VIRTUAL_ENV="/opt/venv"
exec "$@"
