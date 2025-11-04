import os
import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.utils import merge_async_iterators

os.environ["VLLM_SKIP_WARMUP"] = "false"
os.environ['VLLM_CONTIGUOUS_PA'] = 'false'
os.environ['VLLM_MLA_DISABLE_REQUANTIZATION']='1'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES']='true'
os.environ['PT_HPU_WEIGHT_SHARING']='0'
os.environ['VLLM_MLA_PERFORM_MATRIX_ABSORPTION']='0'
os.environ['VLLM_MTP_PRINT_ACCPET_RATE']='0'
os.environ['PT_HPU_LAZY_MODE']='1'
os.environ['VLLM_DELAYED_SAMPLING']='false'
#os.environ['VLLM_USE_V1']='1'

async def main():
    model_name = "/root/.cache/huggingface/GLM-4.5-FP8/"
    engine_args = AsyncEngineArgs(
        model=model_name,
        trust_remote_code=True,
        #enforce_eager=True, # Uncomment this to resolve error
        dtype="bfloat16",
        use_v2_block_manager=True,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        max_model_len=1024,
        num_scheduler_steps=1,
        gpu_memory_utilization=0.5,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        seed=2024
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        generators = []
        for i, prompt in enumerate(prompts):
            generator = llm.generate(prompt, sampling_params, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            if res.finished:
                print(f"result: {res}")

if __name__ == "__main__":
    asyncio.run(main())
