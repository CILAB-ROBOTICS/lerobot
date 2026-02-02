huggingface-cli login --token "$HUGGINGFACE_TOKEN" && \
wandb login "$WANDB_API_KEY" && \
lerobot-train \
  --dataset.repo_id=eunjuri/soft_manip_data \
  --policy.type=act \
  --output_dir=outputs/train/act_soft_manip_data \
  --job_name=act_soft_manip_data \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=eunjuri/hm1_act_policy_soft_manip_data \
  --step=1000 \
  --wandb.project baselines && \
echo "train finished" && \
runpodctl stop pod "$RUNPOD_POD_ID"