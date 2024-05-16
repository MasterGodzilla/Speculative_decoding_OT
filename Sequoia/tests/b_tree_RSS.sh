
# for all models
for model in JackFram/llama-68m JackFram/llama-160m; do
    # for all T values
    for T in 1.0; do
        for f in ../growmaps/k-ary_trees/*; do
            for dataset in cnn openwebtext; do    
                CUDA_VISIBLE_DEVICES=1 python test_specinfer.py --model $model --target ../../../../llama2/llama-2-7b-chat-hf --T $T --P 1.0 --start 0 --end 200 --M 384 --growmap $f --Mode greedy --dataset $dataset >> RSS_log_T1.txt
            done
        done
    done
done