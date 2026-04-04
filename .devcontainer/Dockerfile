# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

# Keep system packages in the image layer so Codespaces prebuilds can cache them.
RUN if [ -d /etc/apt/sources.list.d ]; then \
        grep -RIl 'dl.yarnpkg.com' /etc/apt/sources.list.d | xargs -r rm -f; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libopenmpi-dev \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*
