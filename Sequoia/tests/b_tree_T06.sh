CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/2^2.pt  --Mode mine --dataset cnn >> binary_trees_log_T06.txt
CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/2^2.pt  --Mode mine --dataset openwebtext >> binary_trees_log_T06.txt

CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/2^3.pt  --Mode mine --dataset cnn >> binary_trees_log_T06.txt
CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/2^3.pt  --Mode mine --dataset openwebtext >> binary_trees_log_T06.txt

CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/2^4.pt  --Mode mine --dataset cnn >> binary_trees_log_T06.txt
CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/2^4.pt  --Mode mine --dataset openwebtext >> binary_trees_log_T06.txt

CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/2^5.pt  --Mode mine --dataset cnn >> binary_trees_log_T06.txt
CUDA_VISIBLE_DEVICES=0 python my_testbed.py --model  JackFram/llama-68m   --target ../../../../llama2/llama-2-7b-chat-hf --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../growmaps/k-ary_trees/2^5.pt  --Mode mine --dataset openwebtext >> binary_trees_log_T06.txt


