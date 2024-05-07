python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  \
--T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../binary_tree.pt  --Mode mine --dataset cnn

python my_test_accept.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  \
--T 0.6 --P 1.0  --start 0 --end 200 --M 288 --W 2 \
--ALG stochastic --dataset cnn \
--dst ../hub_accept_rates/llama-68m-7b-chat-T06-P1-cnn.pt 

python test_accept.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  \
--T 0.6 --P 1.0  --start 0 --end 200 --M 288 --W 32 \
--ALG stochastic --dataset cnn \
--dst ../hub_accept_rates/SpecTree-llama-68m-7b-chat-T06-P1-cnn.pt 

CUDA_VISIBLE_DEVICES=1 python my_test_accept.py --model  JackFram/llama-160m   --target ../../../../llama2/llama-2-7b-chat-hf  \
--T 0.6 --P 1.0  --start 0 --end 200 --M 288 --W 2 \
--ALG stochastic --dataset cnn \
--dst ../hub_accept_rates/llama-160m-7b-chat-T06-P1-cnn.pt 

CUDA_VISIBLE_DEVICES=1 python test_accept.py --model  JackFram/llama-160m   --target ../../../../llama2/llama-2-7b-chat-hf  \
--T 0.6 --P 1.0  --start 0 --end 200 --M 288 --W 32 \
--ALG stochastic --dataset cnn \
--dst ../hub_accept_rates/SpecTree-llama-160m-7b-chat-T06-P1-cnn.pt 