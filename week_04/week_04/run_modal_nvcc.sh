#!/bin/bash

set -Eexuao pipefail

modal run ../scripts/modal_nvcc.py --code-path $1