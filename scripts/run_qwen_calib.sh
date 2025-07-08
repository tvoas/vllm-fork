
# MODEL=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# TOKENIZER=$MODEL
# export QUANT_CONFIG=inc_measure.json
# python ./run_example_tp_qwen.py --model $MODEL --tokenizer $TOKENIZER --osl 32 --max_model_len 2048 --max_num_seqs 1 --tp_size 1 --ep_size 1  --inc --dataset pile --nprompts 512

MODEL=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
TOKENIZER=$MODEL
# export QUANT_CONFIG=inc_quant.json
# python ./run_example_tp_qwen.py --model $MODEL --tokenizer $TOKENIZER --osl 32 --max_model_len 2048 --max_num_seqs 1 --tp_size 1 --ep_size 1  --inc  --fp8_kv_cache


# !/bin/bash

if [ $# -gt 0 ] && [ "$1" == "--model_path" ]; then
    model_path=$2
else
    model_path="/mnt/weka/llm/qwen3_pre_release/Qwen3-30B-A3B-250425/"
fi

if [ $# -eq 4 ] && [ "$3" == "--tp_size" ]; then
    tp_size=$4
else
    tp_size=1
fi

export VLLM_LOGGING_LEVEL=DEBUG

model_name=$(basename ${model_path})
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="${model_name}-tp${tp_size}-gsm8k-acc-fp8-${timestamp}"
#limit=None

# quant_file_path="inc_quant.json"
quant_file_path="inc_quant_post.json"
quant_file_path="inc_quant_post_bf16kv.json"
quant_file_path="inc_quant_bf16kv.json"
export VLLM_DISABLE_MARK_SCALES_AS_CONST=1
echo "Eval model ${model_name} with config ${quant_file_path}"

mkdir -p ${output_dir}


# QUANT_CONFIG=${quant_file_path} \
# PT_HPU_LAZY_MODE=1 \
# VLLM_SKIP_WARMUP=true \
# PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
# PT_HPU_WEIGHT_SHARING=0 \
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,quantization=inc,kv_cache_dtype=fp8_inc" \
#   --tasks gsm8k --batch_size 128 --log_samples  --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log_post.txt
  
  
# -fp8 kv before post
# vllm (pretrained=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=1,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,quantization=inc,kv_cache_dtype=fp8_inc), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9325|±  |0.0069|
# |     |       |strict-match    |     5|exact_match|↑  |0.8954|±  |0.0084|

# QUANT_CONFIG=${quant_file_path} \
# PT_HPU_LAZY_MODE=1 \
# VLLM_SKIP_WARMUP=true \
# PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
# PT_HPU_WEIGHT_SHARING=0 \
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True" \
#   --tasks gsm8k --batch_size 128 --log_samples  --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log.txt
  
# -bf16 model
# vllm (pretrained=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=1,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9166|±  |0.0076|
# |     |       |strict-match    |     5|exact_match|↑  |0.8469|±  |0.0099|


# }2025-07-07:12:59:30,249 INFO     [lm_eval.loggers.evaluation_tracker:209] Saving results aggregated
# 2025-07-07:12:59:30,251 INFO     [lm_eval.loggers.evaluation_tracker:290] Saving per-sample results for: gsm8k
# -fp8 kv after post
# vllm (pretrained=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=1,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,quantization=inc,kv_cache_dtype=fp8_inc), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9325|±  |0.0069|
# |     |       |strict-match    |     5|exact_match|↑  |0.8954|±  |0.0084|





# QUANT_CONFIG=${quant_file_path} \
# PT_HPU_LAZY_MODE=1 \
# VLLM_SKIP_WARMUP=true \
# PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
# PT_HPU_WEIGHT_SHARING=0 \
# lm_eval --model vllm \
#   --model_args "pretrained=${model_path},tensor_parallel_size=${tp_size},max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,quantization=inc" \
#   --tasks gsm8k --batch_size 128 --log_samples  --output_path ${output_dir} --show_config 2>&1 | tee ${output_dir}/log_post.txt


# -bf16 model
# vllm (pretrained=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=1,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9166|±  |0.0076|
# |     |       |strict-match    |     5|exact_match|↑  |0.8469|±  |0.0099|

# -bf16 kv before post
# vllm (pretrained=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=1,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,quantization=inc), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9325|±  |0.0069|
# |     |       |strict-match    |     5|exact_match|↑  |0.8954|±  |0.0084|

# -fp8 kv after post
# vllm (pretrained=/mnt/disk9/yiliu7/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=1,max_model_len=4096,max_num_seqs=128,gpu_memory_utilization=0.8,use_v2_block_manager=True,dtype=bfloat16,max_gen_toks=2048,disable_log_stats=True,quantization=inc,kv_cache_dtype=fp8_inc), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 128
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9325|±  |0.0069|
# |     |       |strict-match    |     5|exact_match|↑  |0.8954|±  |0.0084|


