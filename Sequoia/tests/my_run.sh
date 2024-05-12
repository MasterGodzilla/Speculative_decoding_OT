python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../binary_tree.pt  --Mode mine --dataset cnn

CUDA_VISIBLE_DEVICES=3 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../binary_tree.pt  --Mode greedy --dataset cnn

CUDA_VISIBLE_DEVICES=1 python testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../binary_tree.pt  --Mode greedy --dataset cnn

CUDA_VISIBLE_DEVICES=2 python testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../binary_tree.pt  --Mode greedy --dataset cnn

python testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../binary_tree.pt  --Mode baseline --dataset cnn

CUDA_VISIBLE_DEVICES=9 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/68m_7b/growmaps/L40-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn

CUDA_VISIBLE_DEVICES=9 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../L40_growmaps/4-chain.pt  --Mode greedy --dataset cnn
、


python my_testbed.py --model  JackFram/llama-160m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../binary_tree.pt  --Mode mine --dataset cnn

CUDA_VISIBLE_DEVICES=3 python my_testbed.py --model  JackFram/llama-160m   --target ../../../../llama2/llama-2-7b-chat-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../binary_tree.pt  --Mode greedy --dataset cnn