target=meta-llama/Llama-2-13b-chat-hf

for model in JackFram/llama-160m; do
    # for all T values
    for T in 0.6 1.0; do
        # for f in ../growmaps/k-ary_trees/*; do
        for tree in 2^2 2^3 2^4 2^5; do
            for dataset in cnn openwebtext; do    
                CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model $model --target $target \
                --T $T --P 1.0 --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/$tree.pt \
                --Mode mine --dataset $dataset >> SpecHub13b_log.txt
            done
        done
    done
done