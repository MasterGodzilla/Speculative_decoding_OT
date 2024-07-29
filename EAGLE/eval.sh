#export mode=spechub
# for tree in binary2 binary3 binary4 binary5; do
#     for T in 0.6 1.0; do
#         CUDA_VISIBLE_DEVICES=9 python -m eagle.evaluation.gen_ea_answer_vicuna    \
#         --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3    \
#         --base-model-path lmsys/vicuna-7b-v1.3    \
#         --tree-choices $tree    --model-id vicuna-7b-v1.3    \
#         --mode $mode --temperature $T
#     done
# done

for tree in binary2 binary3 binary4 binary5; do
    for T in 0.6 1.0; do
        for mode in spechub RRS RRS_wo_replacement; do
            CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_ea_answer_vicuna    \
            --ea-model-path yuhuili/EAGLE-Vicuna-33B-v1.3    \
            --base-model-path lmsys/vicuna-33b-v1.3   \
            --tree-choices $tree    --model-id vicuna-7b-v1.3    \
            --mode $mode --temperature $T \
            --answer-file mt_bench/33b/$mode$T$tree.txt \
            >> mt_bench/33b.log
        done
    done
done

# CUDA_VISIBLE_DEVICES=9 python -m eagle.evaluation.gen_ea_answer_vicuna    \
#         --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3    \
#         --base-model-path lmsys/vicuna-7b-v1.3    \
#         --tree-choices binary4    --model-id vicuna-7b-v1.3    \
#         --mode spechub --temperature 1.0 --question-end 1 >> spechub_accept_log.txt