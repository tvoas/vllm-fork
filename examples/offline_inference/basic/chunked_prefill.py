import os
import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.utils import merge_async_iterators
from datasets import load_dataset
from pathlib import Path

os.environ["VLLM_SKIP_WARMUP"] = "true"
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
        dtype="bfloat16",
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
        max_model_len=1024*64,
        num_scheduler_steps=1,
        gpu_memory_utilization=0.5,
        max_num_seqs=2,
        max_num_prefill_seqs=1,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
        seed=2024
    )

    dataset_dir = Path("/root/.cache/huggingface/LongBench-v2/")
    num_prompts = 4

    def load_prompts(ds_path: Path, limit: int):
        # Try common file names; adjust as needed.
        candidate_files = [
            "data.jsonl", "dataset.jsonl", "train.jsonl",
            "data.json", "dataset.json", "train.json"
        ]
        data_files = [str(ds_path / f) for f in candidate_files if (ds_path / f).exists()]
        if not data_files:
            raise FileNotFoundError(f"No JSON/JSONL data file found in {ds_path}")
        dataset = load_dataset("json", data_files=data_files, split="train")

        prompts = []
        for ex in dataset:
            # Build MCQ prompt if question + choices present.
            if "question" in ex and "choice_A" in ex:
                parts = [ex["question"]]
                # Optional context
                if isinstance(ex.get("context"), str) and ex["context"].strip():
                    parts.append(f"Context:\n{ex['context'].strip()}\n")
                # Choices
                for label in ("A", "B", "C", "D"):
                    choice_key = f"choice_{label}"
                    if choice_key in ex and isinstance(ex[choice_key], str):
                        parts.append(f"{label}. {ex[choice_key].strip()}")
                # Final answer stub
                parts.append("Answer:")
                prompt = "\n".join(parts)
                prompts.append(prompt)
            else:
                # Fallback to first non-empty text-like field.
                for k in ("prompt", "instruction", "input", "text"):
                    val = ex.get(k)
                    if isinstance(val, str) and val.strip():
                        prompts.append(val.strip())
                        break
            if len(prompts[-1].split()) > 1024 * 32:
                prompts = prompts[:-1]
            if len(prompts) >= limit:
                break
        
        if not prompts:
            raise ValueError("No usable prompts found in dataset.")
        return prompts

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = load_prompts(dataset_dir, num_prompts)

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
