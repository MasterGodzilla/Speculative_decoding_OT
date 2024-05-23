for tree in binary3 binary4 binary5; do
    for mode in spechub RRS RRS_wo_replacement; do
        CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_vicuna    \
        --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3    \
        --base-model-path lmsys/vicuna-7b-v1.3    \
        --tree-choices $tree    --model-id vicuna-7b-v1.3    \
        --mode $mode --temperature 0.6
    done
done


TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_vicuna    \
        --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3    \
        --base-model-path lmsys/vicuna-7b-v1.3    \
        --model-id vicuna-7b-v1.3    \
        --question-end 3 \
        --mode spechub --tree-choices binary4
