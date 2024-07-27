for model in JackFram/llama-68m; do
    # for all T values 0.1, 0.2, ...
    for T in 0.1 0.2 0.3 0.4 0.5 0.7 0.8 0.9; do
        for f in ../growmaps/k-ary_trees/2^5.pt; do
            for dataset in cnn; do    
                CUDA_VISIBLE_DEVICES=1 python testbed.py --model $model --target ../../../../llama2/llama-2-7b-chat-hf --T $T --P 1.0 --start 0 --end 200 --M 384 --growmap $f --Mode greedy --dataset $dataset >> Ts_RRSw.txt
            done
        done
    done
done