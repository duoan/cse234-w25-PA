# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

# Keep system packages in the image layer so Codespaces prebuilds can cache them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libopenmpi-dev \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*
