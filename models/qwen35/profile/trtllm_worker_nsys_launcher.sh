#!/bin/bash
set -Eeo pipefail

# Worker-local replacement for TensorRT-LLM's trtllm-llmapi-launch. The proxy
# command is intentionally not profiled. Nsight is attached only where the
# TensorRT-LLM MPI leader/worker process is created, after rank resolution and
# before CUDA initialization.

task_with_command=("$@")
mpi_rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMI_ID:-0}}}}

log_stderr() { echo -e "\033[33m$*\033[0m" >&2; }
log_stderr "mpi_rank: ${mpi_rank}"

export TLLM_SPAWN_PROXY_PROCESS=1

mpi_world_size() {
    if [ -n "${SLURM_NTASKS:-}" ]; then
        echo "${SLURM_NTASKS}"
    elif [ -n "${OMPI_COMM_WORLD_SIZE:-}" ]; then
        echo "${OMPI_COMM_WORLD_SIZE}"
    else
        echo 1
    fi
}

maybe_export_free_ipc_addr() {
    if [ -n "${TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR:-}" ]; then
        log_stderr "Using user-provided TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR: ${TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR}"
        return
    fi
    TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR=$(
        /usr/bin/python3 -c 'import os,tempfile,uuid; print("ipc://" + os.path.join(tempfile.gettempdir(), "rpc_test_" + str(uuid.uuid4())))'
    )
    export TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR
    log_stderr "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR: ${TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR}"
}

profile_worker() {
    if [ -z "${TLLM_WORKER_NSYS_OUTPUT_DIR:-}" ]; then
        "$@"
        return $?
    fi
    mkdir -p "${TLLM_WORKER_NSYS_OUTPUT_DIR}"
    local mode=${PROFILING_MODE:-worker}
    local output=${TLLM_WORKER_NSYS_OUTPUT_DIR}/${HOSTNAME}-${mode}-rank${mpi_rank}
    log_stderr "Rank${mpi_rank} worker-local Nsight output: ${output}.nsys-rep"
    nsys profile \
        -t cuda,nvtx,ucx \
        --sample=none \
        --cuda-graph-trace=node \
        --trace-fork-before-exec=true \
        -c cudaProfilerApi \
        --capture-range-end=stop \
        --kill=none \
        --wait=all \
        --force-overwrite=true \
        -o "${output}" \
        "$@"
}

export tllm_mpi_size
tllm_mpi_size=$(mpi_world_size)
log_stderr "tllm_mpi_size: ${tllm_mpi_size}"
unset TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY

if [ -z "${mpi_rank}" ] || [ "${mpi_rank}" -eq 0 ]; then
    hmac_key=$(openssl rand -hex 32)
    if [[ ! "${hmac_key}" =~ ^[0-9a-fA-F]{64}$ ]]; then
        log_stderr "Failed to generate a valid TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY"
        exit 1
    fi
    export TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY="${hmac_key}"
    maybe_export_free_ipc_addr

    log_stderr "Rank${mpi_rank} run ${task_with_command[*]} in background"
    mpi_blacklist=(OMPI_ PMIX_ PMI_ SLURM_ MPI_ UCX_ I_MPI_ HYDRA_ KMP_ MPICH_ MV2_ CRAY_)
    (
        for var in $(compgen -e); do
            for prefix in "${mpi_blacklist[@]}"; do
                if [[ "${var}" == "${prefix}"* ]]; then
                    unset "${var}"
                    break
                fi
            done
        done
        set +e
        "${task_with_command[@]}"
        task_exit_code=$?
        log_stderr "Rank${mpi_rank} Task exit code: ${task_exit_code}"
        /usr/bin/python3 -m tensorrt_llm.llmapi.mgmn_leader_node --action stop
        mpi_exit_code=$?
        log_stderr "Rank${mpi_rank} MPI Comm server exit code: ${mpi_exit_code}"
        if [ "${task_exit_code}" -ne 0 ]; then
            exit "${task_exit_code}"
        fi
        exit "${mpi_exit_code}"
    ) &
    subshell_pid=$!

    set +e
    log_stderr "Rank${mpi_rank} run worker-local profiled mgmn leader for world size $(mpi_world_size)"
    profile_worker /usr/bin/python3 -m tensorrt_llm.llmapi.mgmn_leader_node
    mgmn_exit_code=$?
    wait "${subshell_pid}"
    subshell_exit_code=$?
    if [ "${subshell_exit_code}" -ne 0 ]; then
        exit "${subshell_exit_code}"
    fi
    exit "${mgmn_exit_code}"
else
    set +e
    log_stderr "Rank${mpi_rank} run worker-local profiled mgmn worker for world size $(mpi_world_size)"
    profile_worker /usr/bin/python3 -m tensorrt_llm.llmapi.mgmn_worker_node
    exit $?
fi
