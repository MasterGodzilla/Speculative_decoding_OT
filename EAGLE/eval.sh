CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_vicuna\
    --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3\
    --base-model-path lmsys/vicuna-7b-v1.3\
    --tree-choices binary3\
    --model-id vicuna-7b-v1.3\
    --question-end 5