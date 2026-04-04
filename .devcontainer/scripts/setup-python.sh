#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="${workspace_dir}/.venv"
python_bin="${venv_dir}/bin/python"

if [[ ! -d "${venv_dir}" ]]; then
  python3 -m venv "${venv_dir}"
fi

# mpi4py 3.1.5 does not build cleanly with newer setuptools in isolated builds.
"${python_bin}" -m pip install --upgrade "pip<26" "setuptools<80" wheel

tmp_requirements="$(mktemp)"
trap 'rm -f "${tmp_requirements}"' EXIT

for requirements_file in \
  "${workspace_dir}/pa2/requirements.txt" \
  "${workspace_dir}/pa3/requirements.txt"
do
  if [[ -f "${requirements_file}" ]]; then
    grep -hvE '^\s*($|#)|mpi4py==' "${requirements_file}" >> "${tmp_requirements}" || true
  fi
done

if [[ -s "${tmp_requirements}" ]]; then
  sort -u "${tmp_requirements}" -o "${tmp_requirements}"
  "${python_bin}" -m pip install -r "${tmp_requirements}"
fi

MPICC=mpicc "${python_bin}" -m pip install --no-build-isolation "mpi4py==3.1.5"

"${python_bin}" -m pip install \
  ipykernel \
  scikit-learn \
  sentencepiece \
  transformers

# Codespaces in this repo is CPU-only, so install the CPU wheels explicitly.
"${python_bin}" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch \
  torchvision

"${python_bin}" -m ipykernel install \
  --user \
  --name cse234-w25-pa \
  --display-name "Python (.venv)"
