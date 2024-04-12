#CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-stochastic.pt  --Mode greedy >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-stochastic.pt  --Mode greedy --dataset cnn >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-OpenWebText-68m-7b-stochastic.pt  --Mode greedy --dataset openwebtext >> resultsv2.log

#CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-C4-68m-7b-greedy.pt  --Mode greedy >> resultsv2.log                 
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-CNN-68m-7b-greedy.pt  --Mode greedy --dataset cnn >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-7b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_7b/growmaps/A100-OpenWebText-68m-7b-greedy.pt  --Mode greedy --dataset openwebtext >> resultsv2.log

#CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_13b/growmaps/A100-C4-68m-13b-stochastic.pt  --Mode greedy >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_13b/growmaps/A100-CNN-68m-13b-stochastic.pt  --Mode greedy --dataset cnn >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_13b/growmaps/A100-OpenWebText-68m-13b-stochastic.pt  --Mode greedy --dataset openwebtext >> resultsv2.log

#CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_13b/growmaps/A100-C4-68m-13b-greedy.pt  --Mode greedy >> resultsv2.log                 
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_13b/growmaps/A100-CNN-68m-13b-greedy.pt  --Mode greedy --dataset cnn >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-68m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/68m_13b/growmaps/A100-OpenWebText-68m-13b-greedy.pt  --Mode greedy --dataset openwebtext >> resultsv2.log

#CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/160m_13b/growmaps/A100-C4-160m-13b-stochastic.pt  --Mode greedy >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-stochastic.pt  --Mode greedy --dataset cnn >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/160m_13b/growmaps/A100-OpenWebText-160m-13b-stochastic.pt  --Mode greedy --dataset openwebtext >> resultsv2.log

#CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/160m_13b/growmaps/A100-C4-160m-13b-greedy.pt  --Mode greedy >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/160m_13b/growmaps/A100-CNN-160m-13b-greedy.pt  --Mode greedy --dataset cnn >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  JackFram/llama-160m   --target meta-llama/Llama-2-13b-hf  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/160m_13b/growmaps/A100-OpenWebText-160m-13b-greedy.pt  --Mode greedy --dataset openwebtext >> resultsv2.log

#CUDA_VISIBLE_DEVICES=0 python testbed.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-C4-1.3b-33b-stochastic.pt  --Mode greedy >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-CNN-1.3b-33b-stochastic.pt  --Mode greedy --dataset cnn >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/1.3b_33b/growmaps/A100-OpenWebText-1.3b-33b-stochastic.pt  --Mode greedy --dataset openwebtext >> resultsv2.log

#CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/growmaps/1.3b_33b/A100-C4-1.3b-33b-greedy.pt  --Mode greedy >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/growmaps/1.3b_33b/A100-CNN-1.3b-33b-greedy.pt  --Mode greedy --dataset cnn >> resultsv2.log
CUDA_VISIBLE_DEVICES=0 python testbed_greedy.py --model  princeton-nlp/Sheared-LLaMA-1.3B   --target lmsys/vicuna-33b-v1.3  --T 0.6 --P 1.0  --start 0 --end 200 --M 384 --growmap ../A100_growmaps/growmaps/1.3b_33b/A100-OpenWebText-1.3b-33b-greedy.pt  --Mode greedy --dataset openwebtext >> resultsv2.log