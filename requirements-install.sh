#!/bin/bash

usage='Usage: <cpu|cu102|cu113>'
if [[ $# -ne 1 ]]; then
  echo "$0 $usage"
  exit 1
fi
platform="$1"

pip install -r requirements-torch.txt -f "https://download.pytorch.org/whl/$platform/torch_stable.html"
pip install -r requirements.txt
