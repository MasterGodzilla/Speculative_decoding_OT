# for temperature in 0.1 0.5 1.0 1.5 2.0; do 
temperature=1.0
export CUDA_VISIBLE_DEVICES=1
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
export DRAFT_MODEL_REPO=TinyLlama/TinyLlama-1.1B-Chat-v1.0
time python generate.py --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth  --checkpoint_path checkpoints/$MODEL_REPO/model.pth --speculate_k 5 --prompt "Hi my name is" --max_new_tokens 200 --num_samples 5  --temperature $temperature
# done