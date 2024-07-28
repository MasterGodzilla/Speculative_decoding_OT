target=meta-llama/Llama-2-13b-chat-hf
# for model in JackFram/llama-68m JackFram/llama-160m; do
for model in JackFram/llama-160m; do
    # for all T values
    for T in 0.6 1.0; do
        for f in ../growmaps/k-ary_trees/*; do
            for dataset in cnn openwebtext; do    
                CUDA_VISIBLE_DEVICES=0 python test_specinfer.py --model $model --target $target \
                --T $T --P 1.0 --start 0 --end 200 --M 384 --growmap $f \
                --Mode greedy --dataset $dataset >> RRS13b_log.txt
            done
        done
    done
done

for model in JackFram/llama-160m; do
    # for all T values
    for T in 0.6 1.0; do
        for f in ../growmaps/k-ary_trees/*; do
            for dataset in cnn openwebtext; do    
                CUDA_VISIBLE_DEVICES=0 python testbed.py --model $model --target $target \
                --T $T --P 1.0 --start 0 --end 200 --M 384 --growmap $f \
                --Mode greedy --dataset $dataset >> RRSw13b_log.txt
            done
        done
    done
done

for model in JackFram/llama-160m; do
    # for all T values
    for T in 0.6 1.0; do
        for f in ../growmaps/k-ary_trees/*; do
            for dataset in cnn openwebtext; do    
                CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model $model --target $target \
                --T $T --P 1.0 --start 0 --end 200 --M 384 --growmap $f \
                --Mode greedy --dataset $dataset >> RRS13b_log.txt
            done
        done
    done
done