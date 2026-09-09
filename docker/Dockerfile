# No ROS. The Quest stack in manipulation-mono needs ros:jazzy + colcon + Quest2ROS;
# an Xbox pad needs none of it, so this image is ~10x smaller and builds in seconds.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

# ffmpeg  -> imageio writes the mp4s
# libGL/glib -> opencv runtime (headless build still links them)
# iproute2 -> the container assigns 192.168.1.100/24 to the wired NIC
# udev/libudev1 -> evdev enumerating /dev/input
# git -> uv fetches the pinned xArm SDK from a git tag
# xdg-utils -> the dashboard's "open folder" button
# build-essential + linux-libc-dev -> the `evdev` wheel is built from source and needs
#   a C compiler and <linux/input.h>; python:3.12-slim ships neither, and the build
#   fails at `uv pip install` without them.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential linux-libc-dev \
      ffmpeg git iproute2 libgl1 libglib2.0-0 libudev1 udev xdg-utils ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv --python /usr/local/bin/python3 /opt/venv

WORKDIR /workspace
# Dependency layer first, so editing source doesn't re-resolve the world.
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python -e .

# A LOGIN shell (bash -lc) re-initialises PATH from /etc/profile and throws away the
# image's ENV PATH, which silently made every swoosh-* command "not found" inside the
# container -- and exporting PATH from an entrypoint does not survive it either. A
# profile.d drop-in is the one place a login shell will honour.
RUN printf 'export PATH="/opt/venv/bin:$PATH"\nexport VIRTUAL_ENV=/opt/venv\n' \
      > /etc/profile.d/10-venv.sh

COPY docker/entrypoint.bash /entrypoint.bash
RUN chmod +x /entrypoint.bash
ENTRYPOINT ["/entrypoint.bash"]
CMD ["bash"]
