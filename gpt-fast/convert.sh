# karpathy/tinyllamas

# export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
source activate /home/hanchi/miniconda3/envs/gpt-fast
export MODEL_REPO=TinyLlama/TinyLlama-1.1B-Chat-v1.0
python scripts/download.py --repo_id $MODEL_REPO &&  python scripts/convert_tinyllama.py --checkpoint_dir checkpoints/$MODEL_REPO && python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
