#!/usr/bin/env bash
set -xeuo pipefail

project_name='DAPO-FullAsync-Qwen3-30B-NPU'
exp_name='Refactor_fully_async_worker_to_engine'

# Ray
RAY_ADDRESS=http://[fdbd:dc02:29:f28::47]:10446
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# Paths
BYTENAS=${BYTENAS:-"bandai-hl"} #bandai-lf llm-lr-hl  lrm-hl
# Paths
DATA_PATH=${DATA_PATH:-"/mnt/bn/${BYTENAS}"}
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
# MODEL_PATH=/mnt/bn/bandai-hl/shared/models/Qwen3-30B-A3B-Base
# MODEL_PATH=/mnt/bn/bandai-hl/users/lirui.x/lsw_tmp/Qwen3-30B-A3B-Base-merge
# MODEL_PATH=/mnt/bn/bandai-hl/users/lirui.x/lsw_tmp/Qwen3-30B-A3B-Base
MODEL_PATH=/mnt/bn/llm-lr-hl/shared/models/Qwen3-0.6B-Base
CKPTS_DIR=/mnt/bn/bandai-hl/users/lirui.x/lsw_tmp/fully_async_ckpt_1133
TRAIN_FILE=${TRAIN_FILE:-"${DATA_PATH}/shared/data/dapo-math/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_PATH}/shared/data/dapo-math/aime-2024.parquet"}

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=$((1024 * 2))
# max_response_length=$((1024 * 20))
max_response_length=$((1024))
enable_overlong_buffer=False 
# overlong_buffer_len=$((1024 * 4))
overlong_buffer_len=64
overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="token-mean"
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# NNODES=2
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}

# Fully async specific parameters
n_gpus_rollout=16
n_gpus_training=16 
n_nodes_rollout=1
# n_nodes_train=$((NNODES - n_nodes_rollout))
n_nodes_train=1

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*100)))
test_freq=25
staleness_threshold=1.0 #1
# trigger_parameter_sync_step=16
train_bsz=32
total_tain_gpus=$((n_gpus_training * n_nodes_train))
trigger_parameter_sync_step=2
partial_rollout=True #True False
enforce_eager=True
nccl_timeout=7200 #3600

# Performance Related Parameter
sp_size=8 # 8 4
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 2))
ref_offload=True
actor_offload=False
gen_tp=2 #4
fsdp_size=$((n_gpus_training * n_nodes_train)) # 32 64

rollout_is=null
rollout_rs=geometric
rollout_rs_threshold=1.01
rollout_rs_threshold_lower=0.999
rollout_token_veto_threshold=1e-4

# PYTHON_INTERPRETER="/home/hadoop-djst-algoplat/miniconda3/bin/python"
# if [ ! -x "$PYTHON_INTERPRETER" ]; then
#     PYTHON_INTERPRETER="python3"
# fi
    # algorithm.filter_groups.enable=${enable_filter_groups} \
    # algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    # algorithm.filter_groups.metric=${filter_groups_metric} \

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    --address "${RAY_ADDRESS}" \
    -- python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    critic.strategy=fsdp2 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=${nccl_timeout} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    reward_model.reward_manager=dapo \
    trainer.use_legacy_worker_impl=disable \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.test_freq="${test_freq}" \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${n_nodes_train}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${n_nodes_rollout}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq=${test_freq} \
    rollout.total_epochs=10 \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}" \
    trainer.device=npu \
    global_profiler.steps=[1] \
    global_profiler.save_path=/home/tiger/verl/profiler/ \
    global_profiler.tool=npu \
    # async_training.compute_prox_log_prob=True
    # async_training.compute_prox_log_prob=True \
    # algorithm.rollout_correction.rollout_is=${rollout_is} \
    # algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    # algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    # algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    # algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    # actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    # actor_rollout_ref.actor.profiler.enable=true \
    # actor_rollout_ref.actor.profiler.ranks=[0,1] \
    # actor_rollout_ref.actor.profiler.tool_config.npu.contents=[npu,cpu,memory,shapes] \
    # actor_rollout_ref.actor.profiler.tool_config.npu.level='level1' \
    # actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true \
    # actor_rollout_ref.rollout.profiler.enable=true \
    # actor_rollout_ref.ref.profiler.enable=true \
    # actor_rollout_ref.ref.profiler.ranks=[0,1] \
    # actor_rollout_ref.ref.profiler.tool_config.npu.contents=[npu,cpu,memory,shapes] \
    # actor_rollout_ref.ref.profiler.tool_config.npu.level='level1' \
    # actor_rollout_ref.ref.profiler.tool_config.npu.discrete=true \
    # async_training.use_rollout_log_probs=True
        # trainer.use_legacy_worker_impl=disable \

# reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
#     reward_model.overlong_buffer.len=${overlong_buffer_len} \
#     reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \