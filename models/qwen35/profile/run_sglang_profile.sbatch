#!/bin/bash
# Submit with: sbatch --export=ALL,PROFILE_KIND=<attribution|prefill8k|decode-bs32> run_sglang_profile.sbatch
#SBATCH --job-name=qwen35-dep4-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch_long
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --mem=0

set -euo pipefail

: "${PROFILE_KIND:?set PROFILE_KIND to attribution, prefill8k, or decode-bs32}"
case "${PROFILE_KIND}" in
  attribution|prefill8k|decode-bs32) ;;
  *) echo "invalid PROFILE_KIND=${PROFILE_KIND}" >&2; exit 2 ;;
esac

readonly task_root=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/qwen35-agentx-dep4-mtp-profile
readonly source_dir=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/agentx-decode-gap-v2/source/sglang-query-prefill-a31c1e5-production
readonly model_dir=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/kdhruv/models/Qwen3.5-397B-A17B-NVFP4-V2
readonly container_image=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/yangminl/agentx-max-tps-20260810/containers/lmsysorg+sglang+nightly-dev-cu13-20260809-cea16bf2-arm64-enroot.sqsh
readonly source_commit=a31c1e52e947bcbdd0d551c5e2323e96a9bf303b
readonly source_manifest_sha256=4e8e46726a74f357479f2ea251f73913b82f14725c10813d12643712362aff81
readonly model_revision=8f590eae8f10bf55d9a46f79ea0280bde435c9f8
readonly run_dir=${task_root}/runs/sglang/${SLURM_JOB_ID}-${PROFILE_KIND}
readonly cache_dir=${task_root}/cache/${PROFILE_KIND}

mkdir -p "${run_dir}" "${cache_dir}"
chmod 700 "${run_dir}" "${cache_dir}"
exec > >(tee "${run_dir}/job.out") 2> >(tee "${run_dir}/job.err" >&2)

readonly source_manifest=${source_dir}/.agentx-stack-manifest.json
test "$(sha256sum "${source_manifest}" | awk '{print $1}')" = "${source_manifest_sha256}"
manifest_commit=$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["query_prefill_dp_rank_overlap"]["local_commit"])' \
    "${source_manifest}"
)
test "${manifest_commit}" = "${source_commit}"
test -s "${task_root}/producer/profile_workload.py"
test -s "${model_dir}/config.json"
test -s "${container_image}"

echo "job_id=${SLURM_JOB_ID}"
echo "profile_kind=${PROFILE_KIND}"
echo "source_commit=${source_commit}"
echo "source_manifest_sha256=${source_manifest_sha256}"
echo "model_revision=${model_revision}"
echo "run_dir=${run_dir}"

export PROFILE_KIND source_commit source_manifest_sha256 model_revision
srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image="${container_image}" \
  --no-container-entrypoint \
  --no-container-mount-home \
  --container-mounts="${source_dir}:/sgl-workspace/sglang:ro,${model_dir}:/model:ro,${task_root}/producer:/producer:ro,${cache_dir}:/cache,${run_dir}:/run" \
  --container-workdir=/sgl-workspace/sglang \
  bash -lc '
    set -euo pipefail
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/sgl-workspace/sglang/python
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export TRITON_CACHE_DIR=/cache/triton
    export TORCH_EXTENSIONS_DIR=/cache/torch-extensions
    export SGLANG_TORCH_PROFILER_DIR=/run/profiles
    export TMPDIR=/run/tmp
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
    export NCCL_MNNVL_ENABLE=1
    export NCCL_CUMEM_ENABLE=1
    export NCCL_NVLS_ENABLE=0
    export MC_FORCE_MNNVL=1
    export NVSHMEM_REMOTE_TRANSPORT=none
    export SGLANG_ENABLE_FLASHINFER_GEMM=true
    export SGLANG_FLASHINFER_FP4_GEMM_BACKEND=cutlass
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
    export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
    export SGLANG_USE_SYMM_MEM_DP_SYNC=true
    export SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS=4
    export SGLANG_QWEN_DEEPEP_SHARED_OVERLAP=true
    export SGLANG_SIMULATE_ACC_LEN=4.80
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
    export MC_TE_METRIC=true
    export SGLANG_ENABLE_SPEC_V2=1
    export SGLANG_ENABLE_JIT_DEEPGEMM=true
    export SGLANG_MOE_NVFP4_DISPATCH=1
    export SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH=1
    export SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE=1
    export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
    export SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
    export SGLANG_LOG_FORWARD_ITERS=1
    mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$SGLANG_TORCH_PROFILER_DIR" "$TMPDIR"

    graph_args=(--disable-cuda-graph)
    draft_args=(--speculative-moe-runner-backend deep_gemm --speculative-moe-a2a-backend deepep)
    chunk_args=(--chunked-prefill-size 8192 --max-prefill-tokens 8192)
    if [[ "$PROFILE_KIND" == decode-bs32 ]]; then
      graph_args=(--cuda-graph-max-bs 32)
    elif [[ "$PROFILE_KIND" == prefill8k ]]; then
      draft_args=(--speculative-moe-runner-backend triton --speculative-moe-a2a-backend none)
      # SGLang interprets these budgets globally under Attention DP and divides
      # them by dp_size. 32768 / DP4 therefore preserves one full 8192-token
      # owner-rank chunk instead of silently resolving this capture to 2048.
      chunk_args=(--chunked-prefill-size 32768 --max-prefill-tokens 32768)
    fi

    python3 -m sglang.launch_server \
      --model-path /model \
      --served-model-name nvidia/Qwen3.5-397B-A17B-NVFP4 \
      --trust-remote-code \
      --quantization modelopt_fp4 \
      --kv-cache-dtype fp8_e4m3 \
      --tensor-parallel-size 4 \
      --data-parallel-size 4 \
      --expert-parallel-size 4 \
      --enable-dp-attention \
      --enable-dp-lm-head \
      --moe-dense-tp-size 1 \
      --host 127.0.0.1 \
      --port 30000 \
      --context-length 9216 \
      --watchdog-timeout 1800 \
      --mem-fraction-static 0.70 \
      --max-running-requests 128 \
      "${chunk_args[@]}" \
      --disable-radix-cache \
      --page-size 64 \
      --attention-backend trtllm_mha \
      --linear-attn-decode-backend triton \
      --mamba-ssm-dtype bfloat16 \
      --mamba-scheduler-strategy extra_buffer \
      --max-mamba-cache-size 468 \
      --mamba-track-interval 128 \
      --mamba-max-states-per-path 3 \
      --moe-runner-backend flashinfer_cutedsl \
      --moe-a2a-backend flashinfer \
      --disable-shared-experts-fusion \
      --deepep-mode low_latency \
      --ep-dispatch-algorithm static \
      --eplb-algorithm deepseek \
      --speculative-algorithm NEXTN \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 6 \
      --enable-linear-replayssm-spec \
      --linear-replayssm-cache-len 8 \
      --stream-interval 30 \
      --decode-log-interval 10 \
      --enable-layerwise-nvtx-marker \
      --skip-server-warmup \
      "${graph_args[@]}" \
      "${draft_args[@]}" \
      > /run/server.log 2>&1 &
    server_pid=$!

    cleanup() {
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    }
    trap cleanup EXIT

    ready=0
    for _ in $(seq 1 480); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        echo SERVER_EXITED_BEFORE_READY
        tail -n 300 /run/server.log
        exit 1
      fi
      if curl -fsS --max-time 2 http://127.0.0.1:30000/model_info >/run/model_info.json 2>/dev/null; then
        ready=1
        break
      fi
      sleep 5
    done
    if [[ "$ready" != 1 ]]; then
      echo SERVER_READY_TIMEOUT
      tail -n 300 /run/server.log
      exit 1
    fi

    python3 /producer/profile_workload.py \
      --kind "$PROFILE_KIND" \
      --base-url http://127.0.0.1:30000 \
      --output-dir /run/result \
      --expected-ranks 4 \
      --source-commit "$source_commit" \
      --source-manifest-sha256 "$source_manifest_sha256" \
      --model-revision "$model_revision" \
      > /run/workload.log 2>&1
    tail -n 200 /run/workload.log
  '

echo "QWEN35_SGLANG_JOB_OK job_id=${SLURM_JOB_ID} kind=${PROFILE_KIND}"
