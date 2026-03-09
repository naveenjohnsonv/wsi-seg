FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libopenslide0 openslide-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=auto

COPY pyproject.toml README.md /app/
COPY configs /app/configs
COPY src /app/src

RUN uv sync --no-group dev --extra cpu

ENTRYPOINT ["uv", "run", "wsi-seg"]
