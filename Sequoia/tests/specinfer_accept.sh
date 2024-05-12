# for all models
for model in llama-68m llama-160m; do
    # for all T values
    for T in 0.3 0.6 1; do   
        CUDA_VISIBLE_DEVICES=0 python test_accept.py --model JackFram/$model --target ../../../../llama2/llama-2-7b-chat-hf  \
        --T $T --P 1.0  --start 0 --end 200 --M 288 --W 32 \
        --ALG specinfer --dataset cnn \
        --dst ../hub_accept_rates/SpecInferTree-$model-7b-chat-T$T-P1-cnn.pt
    done
done