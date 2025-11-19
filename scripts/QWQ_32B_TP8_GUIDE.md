Is worker applying patch again, causing our model request to grow but chunked code in runner appends more, which means we grow more and more and more? Need to verify

Stream in N>1 chunks while N=0 runs. Eliminate gap.

Multistep?

# Qwen3 with PP=2 Guide

This guide provides instructions to build, run, and configure a Docker container for vLLM, as well as perform accuracy and throughput benchmarking on this. Uses best known configs.

## Setup the Docker Container on Host

### Step 1: Set Environment Variables

Set the following environment variables for the Docker setup:

sudo hl-smi --format csv -Q bus_id,module_id,index | awk -F',' 'NR>1 {gsub(/ /,""); print $1}' | while read id; do sudo hl-smi -r -i "$id" & done

```bash
export BUILT_IMAGE="vllm-hpu-env"
export CONTAINER_NAME="vllm-node"
export MAPPED_MODEL_PATHS="/mnt/disk1/hf_models/"
export no_proxy="localhost,127.0.0.1,10.239.129.9"
export http_proxy="http://proxy.ims.intel.com:911"
export https_proxy="http://proxy.ims.intel.com:911"
```

### Step 2: Build the Docker Image
Run the following commands to build the Docker image:

```bash
git clone -b "dev/aice/v1.22.0/pp_tvoas" https://github.com/tvoas/vllm-fork.git
cd vllm-fork
sudo docker build -f docker/Dockerfile.hpu \
  --build-arg http_proxy=${http_proxy} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg ftp_proxy=${ftp_proxy} \
  --build-arg no_proxy=${no_proxy} \
  -t ${BUILT_IMAGE} .
```

### Step 3: Start the Docker Container
Run the following commands to start the Docker container in the background:

```bash
cd tvoas
rm -rf vllm-fork
git clone -b "dev/aice/v1.22.0/pp_tvoas" https://github.com/tvoas/vllm-fork.git
cd vllm-fork
git clone  -b "aice/v1.22.0" https://github.com/HabanaAI/vllm-hpu-extension.git
git fetch origin; git reset --hard  de21efbdc0ba74828e48cab7d0820278070d83db; git checkout origin/dev/aice/v1.22.0/pp_tvoas_chunk_debug scripts/start_gaudi_vllm_server.sh scripts/utils.sh
sudo docker build -f docker/Dockerfile.hpu \
  --build-arg http_proxy=${http_proxy} \
  --build-arg https_proxy=${https_proxy} \
  --build-arg ftp_proxy=${ftp_proxy} \
  --build-arg no_proxy=${no_proxy} \
  -t ${BUILT_IMAGE} .

sudo docker kill ${CONTAINER_NAME}
sudo docker rm ${CONTAINER_NAME}

sudo docker run -td \
  --entrypoint /bin/bash \
  --network host \
  --ipc=host \
  --name ${CONTAINER_NAME} \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add SYS_NICE \
  --privileged \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e PT_HPU_ENABLE_LAZY_COLLECTIVES=1 \
  -e PT_HPU_LAZY_MODE=1 \
  -e LD_LIBRARY_PATH=/root/libfabric/lib:/opt/amazon/openmpi/lib:/usr/lib/habanalabs \
  -v ${MAPPED_MODEL_PATHS}:/root/.cache/huggingface \
  ${BUILT_IMAGE} \
  -c "tail -f /dev/null"
rm ../vllm_hpu_node.tar
sudo docker save ${BUILT_IMAGE} > ../vllm_hpu_node.tar
```

### Step 4: Access the Docker Container
Run the following command to connect to the running container:

```bash
sudo docker exec -it ${CONTAINER_NAME} bash
```

## Setup within the Container

After entering the Docker container using `docker exec`, follow these steps to configure the environment. The setup is divided into three sections: general setup (always required), HCCL/libfabric setup (only required if using HCCL as the communication backend), and vLLM server settings.

---

### Step 1: General Setup (Always Required)

Run the following commands to set up proxies, install required libraries, and configure the environment:

```bash
export no_proxy=localhost,127.0.0.1,10.239.129.9
export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911
```

### Step 2: HCCL and Libfabric Setup (Optional)
```Note:``` This section is only required if using HCCL as the communication backend. If using GLOO, you can skip this section.

Run the following commands to set up HCCL and libfabric:

```bash
export REQUIRED_VERSION=1.22.0

# Download and build libfabric
wget https://github.com/ofiwg/libfabric/releases/download/v$REQUIRED_VERSION/libfabric-$REQUIRED_VERSION.tar.bz2 -P /tmp/libfabric
pushd /tmp/libfabric
tar -xf libfabric-$REQUIRED_VERSION.tar.bz2
export LIBFABRIC_ROOT=$HOME/libfabric
mkdir -p ${LIBFABRIC_ROOT}
chmod 777 ${LIBFABRIC_ROOT}
cd libfabric-$REQUIRED_VERSION/
./configure --prefix=$LIBFABRIC_ROOT --with-synapseai=/usr
make -j 32 && make install
popd
rm -rf /tmp/libfabric
export LD_LIBRARY_PATH=$LIBFABRIC_ROOT/lib:$LD_LIBRARY_PATH

# Install libfabric utilities
#apt update && apt install -y libfabric-bin
#fi_info --version

# Clone and build HCCL OFI wrapper
git clone https://github.com/HabanaAI/hccl_ofi_wrapper.git
cd hccl_ofi_wrapper
export LIBFABRIC_ROOT=$HOME/libfabric/
make
cp libhccl_ofi_wrapper.so /usr/lib/habanalabs/libhccl_ofi_wrapper.so
ldconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/habanalabs/

cd ..
```

### Step 3: Set Environment Variables

Run the following commands to configure the environment:

```bash
export MAX_MODEL_LEN=65536
export MAX_NUM_BATCHED_TOKENS=65536
export MAX_NUM_SEQS_PER_PP_GROUP=2
export VLLM_FP32_SOFTMAX=false
export HABANA_VISIBLE_DEVICES=ALL
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export VLLM_RAY_DISABLE_LOG_TO_DRIVER=1
export RAY_IGNORE_UNHANDLED_ERRORS=1
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES=0,1,2,3,4,5,6,7
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_GRAPH_RESERVED_MEM=0.3
export VLLM_GRAPH_PROMPT_RATIO=0
export VLLM_EP_SIZE=4
export VLLM_PP_USE_CPU_COMS=1
export PT_HPU_RECIPE_CACHE_CONFIG=/data/40960_cache,false,40960
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=2
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=128
export VLLM_PROMPT_SEQ_BUCKET_MAX=65536
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=1
export VLLM_DECODE_BS_BUCKET_MAX=2
export VLLM_DECODE_BLOCK_BUCKET_MIN=64
export VLLM_DECODE_BLOCK_BUCKET_STEP=64
export VLLM_DECODE_BLOCK_BUCKET_MAX=1024
export VLLM_SKIP_WARMUP=false
export VLLM_DELAYED_SAMPLING=false
export QUANT_CONFIG=/workspace/vllm-hpu-extension/calibration/quantization/glm-4.5-air-fp8-g2/maxabs_quant_g2.json
export VLLM_DISABLE_MARK_SCALES_AS_CONST=false
```

---

### Notes:
- The General Setup section is mandatory for all configurations.
- The HCCL and Libfabric Setup section is only required if using HCCL as the communication backend. If using GLOO, you can skip this section entirely.
- Ensure that all commands are executed inside the Docker container after running docker exec.
- Ensure the MAPPED_MODEL_PATHS directory contains the required model data.
- The container will remain running in the background and can be accessed anytime using docker exec.

## Offline Benchmarking Guide for vLLM

This guide provides instructions to configure the environment and run offline performance benchmarking for vLLM.

### Step 1: Run the Offline Benchmark
Run the following command to execute the offline throughput benchmark. This runs input 8K, output 2K, concurrency 48, 240 total samples:

```bash
export QUANT_CONFIG=/workspace/vllm/scripts/glm_pp/quant_configs/inc_measure.json
export INC_ENABLE_TP_RANK_INFO=1
VLLM_SKIP_WARMUP=true VLLM_DISABLE_MARK_SCALES_AS_CONST=1 VLLM_HPU_CONVERT_TO_FP8UZ=0 python3 vllm/benchmarks/benchmark_throughput.py \
  --backend vllm \
  --model /root/.cache/huggingface/GLM-4.5-FP8-G2 \
  --tokenizer /root/.cache/huggingface/GLM-4.5-FP8-G2 \
  --device hpu \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS_PER_PP_GROUP} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --use-padding-aware-scheduling \
  --use-v2-block-manager \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --num-scheduler-steps 1 \
  --gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
  --dataset-name pile10k \
  --dataset-path /root/.cache/huggingface/pile-10k/ \
  --num-prompts 512 \
  --input-len 8192 \
  --output-len 2048 \
  --async-engine \
  --seed 0
```

### Step 2: Expected performance

```bash
Throughput: 0.12 requests/s, 1135.31 total tokens/s, 243.48 output tokens/s
Total num prompt tokens:  1800362
Total num output tokens:  491520
```

---

### Notes:
- Ensure the dataset file (sonnet.txt) is available in the specified path.
- Adjust the numactl settings (-C and -m) based on your system's CPU and memory configuration.
- This configuration is optimized for performance benchmarking and does not include accuracy evaluation, as offline benchmarks focus solely on throughput.

# Online Benchmarking Guide for vLLM

This guide provides instructions to configure the environment, start the server, and run online performance and accuracy benchmarking for vLLM.

---

### Step 1: Start the Server
Run the following command to start the vLLM server:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8688 \
  --block-size 128 \
  --model /root/.cache/huggingface/GLM-4.5-FP8/ \
  --device hpu \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS_PER_PP_GROUP} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --disable-log-requests \
  --use-padding-aware-scheduling False \
  --distributed_executor_backend mp \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --enable-chunked-prefill \
  --num-scheduler-steps 1 \
  --tool-call-parser glm45 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION}



  
--max-model-len ${MAX_MODEL_LEN} --max-num-seqs ${MAX_NUM_SEQS_PER_PP_GROUP} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION}
```

### Step 2: Run Online Throughput Benchmark
Run the following command to execute the online throughput benchmark. This runs input 8K, output 2K, concurrency 48, 240 total samples:

```NOTE:``` these client side commands for performance and accuracy should be run in a second ```docker exec``` terminal. Ensure proxies are also set in this new terminal.

```bash
python3 vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model /root/.cache/huggingface/GLM-4.5-Air-FP8-G2/ \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8688 \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --sonnet-input-len 8192 \
  --sonnet-output-len 2048 \
  --max-concurrency 4 \
  --num-prompts 8 \
  --request-rate inf \
  --seed 0 \
  --ignore-eos \
  --save-result \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 25,50,75,90,95,99 \
  --result-filename online_enchmark_results.json
```

### Step 3: Expected performance


```bash
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 48
============ Serving Benchmark Result ============
Successful requests:                     144
Benchmark duration (s):                  1094.67
Total input tokens:                      1080086
Total generated tokens:                  294912
Request throughput (req/s):              0.13
Output token throughput (tok/s):         269.41
Total Token throughput (tok/s):          1256.09
---------------Time to First Token----------------
Mean TTFT (ms):                          55026.53
Median TTFT (ms):                        55787.16
P25 TTFT (ms):                           32196.89
P50 TTFT (ms):                           55787.16
P75 TTFT (ms):                           79615.93
P90 TTFT (ms):                           92815.00
P95 TTFT (ms):                           97281.31
P99 TTFT (ms):                           102686.59
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          151.37
Median TPOT (ms):                        151.81
P25 TPOT (ms):                           140.17
P50 TPOT (ms):                           151.81
P75 TPOT (ms):                           163.37
P90 TPOT (ms):                           169.84
P95 TPOT (ms):                           172.11
P99 TPOT (ms):                           174.02
---------------Inter-token Latency----------------
Mean ITL (ms):                           151.37
Median ITL (ms):                         128.68
P25 ITL (ms):                            123.40
P50 ITL (ms):                            128.68
P75 ITL (ms):                            132.67
P90 ITL (ms):                            135.98
P95 ITL (ms):                            137.22
P99 ITL (ms):                            145.40
----------------End-to-end Latency----------------
Mean E2EL (ms):                          364871.45
Median E2EL (ms):                        362787.96
P25 E2EL (ms):                           362644.62
P50 E2EL (ms):                           362787.96
P75 E2EL (ms):                           369201.48
P90 E2EL (ms):                           369265.61
P95 E2EL (ms):                           369269.04
P99 E2EL (ms):                           369271.81
==================================================
```

### Step 4: Run LM-Eval Accuracy Benchmark
Run the following command to evaluate accuracy using lm-eval. This runs full humaneval dataset and 1000 samples of gsm8k (full dataset gets asyncio errors):

```bash
clear

echo "/root/.cache/huggingface/garb_g2d_1220_271_tv_new_loop_1ve_glm45air_no_inc_warmup_lazy_2048_1pref0mix_chunk/"
export HF_ALLOW_CODE_EVAL=1; export LM_ADDR=127.0.0.1; export LM_PORT=8688; export LM_MODEL=/root/.cache/huggingface/GLM-4.5-Air-FP8-G2; export LM_BATCH_SIZE=1; export LM_LIMIT=16
export LM_CONCURRENCY=1; python vllm/benchmarks/benchmark_serving.py --backend vllm --model ${LM_MODEL} --trust-remote-code --host ${LM_ADDR} --port ${LM_PORT} --dataset-name random --random-input-len 30151 --random-output-len 230 --random-range-ratio 0.8 --num-prompts 8 --request-rate inf --seed 0 --ignore_eos --max-concurrency ${LM_CONCURRENCY} --percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 10,90,99
sleep 60
export LM_CONCURRENCY=2; python vllm/benchmarks/benchmark_serving.py --backend vllm --model ${LM_MODEL} --trust-remote-code --host ${LM_ADDR} --port ${LM_PORT} --dataset-name random --random-input-len 30151 --random-output-len 230 --random-range-ratio 0.8 --num-prompts 16 --request-rate inf --seed 0 --ignore_eos --max-concurrency ${LM_CONCURRENCY} --percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 10,90,99
sleep 60
export LM_CONCURRENCY=4; python vllm/benchmarks/benchmark_serving.py --backend vllm --model ${LM_MODEL} --trust-remote-code --host ${LM_ADDR} --port ${LM_PORT} --dataset-name random --random-input-len 30151 --random-output-len 230 --random-range-ratio 0.8 --num-prompts 32 --request-rate inf --seed 0 --ignore_eos --max-concurrency ${LM_CONCURRENCY} --percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 10,90,99
sleep 60
export LM_CONCURRENCY=4; lm_eval --model local-completions \
  --tasks gsm8k,humaneval \
  --model_args model=$LM_MODEL,base_url=http://$LM_ADDR:$LM_PORT/v1/completions,num_concurrent=$LM_CONCURRENCY,trust_remote_code=True \
  --batch_size $LM_BATCH_SIZE \
  --confirm_run_unsafe_code \
  --log_samples \
  --limit $LM_LIMIT \
  --output_path lm_eval_results.json


export HF_ALLOW_CODE_EVAL=1; export LM_ADDR=127.0.0.1; export LM_PORT=8688; export LM_MODEL=/root/.cache/huggingface/GLM-4.5-Air-FP8-G2; export LM_BATCH_SIZE=1; export LM_LIMIT=16
export LM_CONCURRENCY=4; python vllm/benchmarks/benchmark_serving.py --backend vllm --model ${LM_MODEL} --trust-remote-code --host ${LM_ADDR} --port ${LM_PORT} --dataset-name random --random-input-len 30151 --random-output-len 230 --random-range-ratio 0.8 --num-prompts 8 --request-rate inf --seed 0 --ignore_eos --max-concurrency ${LM_CONCURRENCY} --percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 10,90,99
sleep 60
export LM_CONCURRENCY=2; python vllm/benchmarks/benchmark_serving.py --backend vllm --model ${LM_MODEL} --trust-remote-code --host ${LM_ADDR} --port ${LM_PORT} --dataset-name random --random-input-len 30151 --random-output-len 230 --random-range-ratio 0.8 --num-prompts 16 --request-rate inf --seed 0 --ignore_eos --max-concurrency ${LM_CONCURRENCY} --percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 10,90,99
sleep 60
export LM_CONCURRENCY=4; python vllm/benchmarks/benchmark_serving.py --backend vllm --model ${LM_MODEL} --trust-remote-code --host ${LM_ADDR} --port ${LM_PORT} --dataset-name random --random-input-len 30151 --random-output-len 230 --random-range-ratio 0.8 --num-prompts 32 --request-rate inf --seed 0 --ignore_eos --max-concurrency ${LM_CONCURRENCY} --percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 10,90,99
sleep 60
export LM_CONCURRENCY=4; lm_eval --model local-completions \
  --tasks gsm8k,humaneval \
  --model_args model=$LM_MODEL,base_url=http://$LM_ADDR:$LM_PORT/v1/completions,num_concurrent=$LM_CONCURRENCY,trust_remote_code=True \
  --batch_size $LM_BATCH_SIZE \
  --confirm_run_unsafe_code \
  --log_samples \
  --limit $LM_LIMIT \
  --output_path lm_eval_results.json


export HF_ALLOW_CODE_EVAL=1; export LM_ADDR=127.0.0.1; export LM_PORT=8688; export LM_MODEL=/root/.cache/huggingface/GLM-4.5-FP8; export LM_BATCH_SIZE=1; export LM_LIMIT=16
export LM_CONCURRENCY=1; python vllm/benchmarks/benchmark_serving.py --backend vllm --model ${LM_MODEL} --trust-remote-code --host ${LM_ADDR} --port ${LM_PORT} --dataset-name random --random-input-len 10000 --random-output-len 8 --random-range-ratio 0.8 --num-prompts 2 --request-rate inf --seed 0 --ignore_eos --max-concurrency ${LM_CONCURRENCY} --percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 10,90,99


export LM_CONCURRENCY=4
python vllm/benchmarks/benchmark_serving.py --backend vllm --model ${LM_MODEL} --trust-remote-code --host ${LM_ADDR} --port ${LM_PORT} --dataset-name random --random-input-len 30151 --random-output-len 230 --random-range-ratio 0.8 --num-prompts ${LM_CONCURRENCY}0 --request-rate inf --seed 0 --ignore_eos --max-concurrency ${LM_CONCURRENCY} --percentile-metrics ttft,tpot,itl,e2el  --metric-percentiles 10,99
lm_eval --model local-completions \
  --tasks gsm8k,humaneval \
  --model_args model=$LM_MODEL,base_url=http://$LM_ADDR:$LM_PORT/v1/completions,num_concurrent=$LM_CONCURRENCY,trust_remote_code=True \
  --batch_size $LM_BATCH_SIZE \
  --confirm_run_unsafe_code \
  --log_samples \
  --limit $LM_LIMIT \
  --output_path lm_eval_results.json











lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args model=$LM_MODEL,base_url=http://$LM_ADDR:$LM_PORT/v1/chat/completions,num_concurrent=$LM_CONCURRENCY,trust_remote_code=True \
  --batch_size $LM_BATCH_SIZE \
  --confirm_run_unsafe_code \
  --log_samples \
  --apply_chat_template \
  --limit $LM_LIMIT \
  --output_path lm_eval_results.json
```

### Step 5: Expected accuracy

```bash
local-completions (model=/root/.cache/huggingface/QwQ-32B/,base_url=http://127.0.0.1:8688/v1/completions,num_concurrent=48,trust_remote_code=True), gen_kwargs: (None), limit: 800.0, num_fewshot: None, batch_size: 1
|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k    |      3|flexible-extract|     5|exact_match|↑  |0.7800|±  |0.0147|
|         |       |strict-match    |     5|exact_match|↑  |0.8275|±  |0.0134|
|humaneval|      1|create_test     |     0|pass@1     |   |0.5244|±  |0.0391|
```

---

### Notes:
- Ensure the dataset file (sonnet.txt) is available in the specified path for throughput benchmarking.
- Adjust the numactl settings (-C and -m) based on your system's CPU and memory configuration.