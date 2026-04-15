#!/bin/bash
cd /home/agent/work/msa
source .venv/bin/activate
export MASTER_PORT=${MASTER_PORT:-29513}
export MSA_PORT=${MSA_PORT:-8378}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec python3 server.py >> /home/agent/work/msa/server.log 2>&1
