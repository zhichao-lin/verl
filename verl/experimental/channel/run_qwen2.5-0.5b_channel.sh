set -xeuo pipefail

MODEL_PATH="/model/Qwen2.5-0.5B-Instruct"
TRAIN_FILE="/data/gsm8k/train.parquet"
TEST_FILE="/data/gsm8k/test.parquet"

log_dir="./logs"
mkdir -p ${log_dir}
timestamp=$(date +"%Y%m%d%H%M%S")
log_file="${log_dir}/qwen2.5-0.5b_channel_${timestamp}.log"

# ASCEND specific settings
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_LOGGING_LEVEL=DEBUG
export RLINF_LOG_LEVEL=DEBUG

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

python3 -m verl.experimental.channel.main_ppo \
    --config-name='channel_ppo_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts_workers=1 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2.5_0.5b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1000 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=2 \
    trainer.val_before_train=False \
    2>&1 | tee "$log_file"
echo "Finished, log is saved in: $log_file"