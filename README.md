# SpecHub: Provable Acceleration to Multi-Draft Speculative Decoding

Welcome to the SpecHub repository! This repository contains the implementation of SpecHub, a novel approach to accelerating the inference process of Large Language Models (LLMs) through an optimized speculative decoding framework.

## Overview

SpecHub addresses the inefficiencies of traditional Multi-Draft Speculative Decoding (MDSD) methods by optimizing the acceptance rate of draft tokens using an Optimal Transport (OT) formulation. By strategically designing the joint distribution of draft tokens and the accepted token, SpecHub significantly accelerates the decoding process and achieves higher acceptance rates compared to existing methods.

## Key Features

- **Improved Efficiency**: SpecHub enhances batch efficiency, generating 0.05-0.27 more tokens per step than Recursive Rejection Sampling (RRS) and achieves equivalent batch efficiency with half the concurrency.
- **Optimal Transport Formulation**: Utilizes a simplified Linear Programming (LP) model to optimize the acceptance rate of draft tokens.
- **Seamless Integration**: Can be integrated into existing MDSD frameworks with minimal computational overhead.

## Usage

To use SpecHub in your projects, follow the steps below:

### Testing Llama-2 vs Draft Model

First, install relevant packages.

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.2
pip install accelerate==0.26.1
pip install datasets==2.16.1
pip install einops
pip install protobuf
pip install sentencepiece
pip install typing-extensions
```

Set up your Huggingface Access Tokens by running the following command:

```bash
huggingface-cli login
```


Then, run the following command to test the Llama-2 model with the draft model:

```bash
cd Sequoia/tests
# SpecHub
echo "SpecHub" >> ../../log.txt
python my_testbed.py --model meta-llama/Llama-2-7b-chat-hf \
--target JackFram/llama-68m \
--T 1.0 --P 1.0 --start 0 --end 200 --M 384 \
--growmap ../growmaps/k-ary_trees/2^5.pt \
--Mode mine --dataset cnn >> ../../log.txt

# RRS
echo "RRS" >> ../../log.txt
python test_specinfer.py --model meta-llama/Llama-2-7b-chat-hf \
--target JackFram/llama-68m \
--T 1.0 --P 1.0 --start 0 --end 200 --M 384 \
--growmap ../growmaps/k-ary_trees/2^5.pt \
--Mode greedy --dataset cnn >> ../../log.txt

# RRSw
echo "RRSw" >> ../../log.txt
python testbed.py --model meta-llama/Llama-2-7b-chat-hf \
--target JackFram/llama-68m \
--T 1.0 --P 1.0 --start 0 --end 200 --M 384 \
--growmap ../growmaps/k-ary_trees/2^5.pt \
--Mode greedy --dataset cnn >> ../../log.txt
```
