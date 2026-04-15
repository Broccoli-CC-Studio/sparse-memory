#!/bin/bash
cd /home/agent/work/msa
export MASTER_PORT=${MASTER_PORT:-29513}
export MSA_PORT=${MSA_PORT:-8379}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
exec uv run python3 server.py >> /home/agent/work/msa/server.log 2>&1
