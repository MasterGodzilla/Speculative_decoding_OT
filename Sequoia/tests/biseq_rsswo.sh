# for all models
for model in 160m; do
    # for all T values
    for T in 0.6 1; do
        for f in ../growmaps/biseq_trees/llama-7b-$model/T$T/*; do
            for dataset in cnn openwebtext; do    
                CUDA_VISIBLE_DEVICES=1 python testbed.py --model JackFram/llama-$model \
                --target ../../../../llama2/llama-2-7b-chat-hf --T $T --P 1.0 --start 0 --end 200 --M 384 \
                --growmap $f --Mode greedy --dataset $dataset >> RSS_wo_log.txt
            done
        done
    done
done