#!/bin/bash
set -e
python -m pip install -q -r requirements.txt
python -m sltrain.train "$@"
