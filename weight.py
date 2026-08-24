import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WEIGHT_KEYS = [
    "peak_similarity",
    "slope_abs",
    "rising_slope",
    "falling_slope",
    "boundary_change",
    "context_density",
]

SYSTEM_PROMPT = """You are an expert in video analysis. Given a question about a video, analyze what kinds of visual evidence are needed to answer it, then output a weight vector for controlling frame selection from a query-relevance curve.

Weight dimensions:
- peak_similarity: Whether frames highly similar to the question are needed. This is high for descriptive questions such as "What is in the scene?"
- slope_abs: Whether frames with sharp changes are needed. This is high for dynamic or process-oriented questions.
- rising_slope: Whether frames from increasing relevance regions are needed. This is high for causal or setup questions such as "Why?"
- falling_slope: Whether frames from decreasing relevance regions are needed. This is high for follow-up or consequence questions such as "What happens after that?"
- boundary_change: Whether event boundaries or abrupt transition frames are needed. This is high for questions about turning points.
- context_density: Whether information-dense temporal regions are needed. This is high for questions requiring broader context.

Return strictly in the following JSON format. Each weight must be an integer from 0 to 10:
{
  "peak_similarity": <int>,
  "slope_abs": <int>,
  "rising_slope": <int>,
  "falling_slope": <int>,
  "boundary_change": <int>,
  "context_density": <int>,
  "reasoning": "<brief explanation>"
}

Return only JSON. Do not include any other text."""

USER_TEMPLATE = """Question: "{question}"

Analyze the visual evidence required by this question and return the JSON weight vector. Return only JSON."""


def extract_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    for start in [i for i, ch in enumerate(stripped) if ch == "{"]:
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    return None


def build_default_result(question: str) -> dict[str, Any]:
    return {
        "peak_similarity": 7,
        "slope_abs": 5,
        "rising_slope": 5,
        "falling_slope": 5,
        "boundary_change": 5,
        "context_density": 5,
        "reasoning": f"Parsing failed. Default weights were used for question: {question[:80]}",
    }


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    for key in WEIGHT_KEYS:
        value = result.get(key, 5)
        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, (int, float)):
            value = int(value)
        else:
            value = 5
        result[key] = max(0, min(10, value))

    reasoning = result.get("reasoning", "")
    result["reasoning"] = reasoning if isinstance(reasoning, str) else str(reasoning)
    return result


def format_question(item: dict[str, Any], append_options: bool) -> str:
    question = str(item["question"])

    if not append_options:
        return question

    options = item.get("options")
    if not options:
        return question

    if isinstance(options, list):
        options_text = " ".join(str(option) for option in options)
    elif isinstance(options, dict):
        options_text = " ".join(f"{key}: {value}" for key, value in options.items())
    else:
        options_text = str(options)

    return f"{question}\nOptions: {options_text}"


def get_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def run_worker(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    tag = f"[Worker {args.worker_id} | GPU {args.gpu_id}]"

    with open(args.shard_file, "r", encoding="utf-8") as f:
        shard_data = json.load(f)

    print(f"{tag} started with {len(shard_data)} questions", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = get_torch_dtype(args.dtype)

    print(f"{tag} loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    print(f"{tag} model loaded", flush=True)

    results: list[dict[str, Any]] = []
    failed_count = 0
    total = len(shard_data)
    start_time = time.time()

    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch_items = shard_data[batch_start:batch_end]

        batch_texts = []
        for item in batch_items:
            full_question = format_question(item, args.append_options)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(question=full_question)},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            batch_texts.append(text)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]

        for output, item in zip(outputs, batch_items):
            response = tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True,
            ).strip()

            parsed = extract_json(response)
            if parsed is None:
                result = build_default_result(str(item["question"]))
                failed_count += 1
                print(
                    f"{tag} failed to parse question {item.get('question_id', '?')}",
                    flush=True,
                )
            else:
                result = validate_result(parsed)

            results.append(
                {
                    "question_id": item.get("question_id", ""),
                    "video_id": item.get("video_id", ""),
                    "videoID": item.get("videoID", ""),
                    "question": item["question"],
                    "task_type": item.get("task_type", ""),
                    "peak_similarity": result["peak_similarity"],
                    "slope_abs": result["slope_abs"],
                    "rising_slope": result["rising_slope"],
                    "falling_slope": result["falling_slope"],
                    "boundary_change": result["boundary_change"],
                    "context_density": result["context_density"],
                    "reasoning": result["reasoning"],
                }
            )

        done = batch_end
        elapsed = time.time() - start_time
        speed = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / speed if speed > 0 else 0.0

        print(
            f"{tag} progress {done}/{total} "
            f"({done / total * 100:.1f}%) | {speed:.2f} q/s | ETA {eta:.0f}s",
            flush=True,
        )

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - start_time
    print(
        f"{tag} finished: {total} questions, {failed_count} parse failures, "
        f"{elapsed_total:.1f}s",
        flush=True,
    )


def parse_gpu_ids(gpus: str) -> list[str]:
    gpu_ids = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided.")
    return gpu_ids


def split_shards(data: list[dict[str, Any]], num_workers: int) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    for index, item in enumerate(data):
        shards[index % num_workers].append(item)
    return shards


def run_scheduler(args: argparse.Namespace) -> None:
    gpu_ids = parse_gpu_ids(args.gpus)
    num_workers = len(gpu_ids) * args.workers_per_gpu

    print("=" * 72)
    print("Video QA weight generation")
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"GPUs: {gpu_ids}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print(f"Total workers: {num_workers}")
    print(f"Batch size per worker: {args.batch_size}")
    print(f"Append options: {args.append_options}")
    print("=" * 72)

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON list.")

    total = len(data)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shards = split_shards(data, num_workers)
    shard_input_files: list[Path] = []
    shard_output_files: list[Path] = []

    for worker_id, shard in enumerate(shards):
        shard_input = tmp_dir / f"input_{worker_id}.json"
        shard_output = tmp_dir / f"output_{worker_id}.json"

        with open(shard_input, "w", encoding="utf-8") as f:
            json.dump(shard, f, ensure_ascii=False)

        shard_input_files.append(shard_input)
        shard_output_files.append(shard_output)

        gpu_id = gpu_ids[worker_id // args.workers_per_gpu]
        print(f"Worker {worker_id} | GPU {gpu_id} | {len(shard)} examples")

    script_path = Path(__file__).resolve()
    processes: list[subprocess.Popen[Any]] = []

    start_time = time.time()

    for worker_id in range(num_workers):
        gpu_id = gpu_ids[worker_id // args.workers_per_gpu]

        command = [
            sys.executable,
            str(script_path),
            "--worker",
            "--worker-id",
            str(worker_id),
            "--gpu-id",
            str(gpu_id),
            "--model-path",
            args.model_path,
            "--shard-file",
            str(shard_input_files[worker_id]),
            "--output-file",
            str(shard_output_files[worker_id]),
            "--batch-size",
            str(args.batch_size),
            "--max-input-tokens",
            str(args.max_input_tokens),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--dtype",
            args.dtype,
        ]

        if args.append_options:
            command.append("--append-options")
        if args.trust_remote_code:
            command.append("--trust-remote-code")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        process = subprocess.Popen(command, env=env)
        processes.append(process)

        print(f"Started worker {worker_id} on GPU {gpu_id}, PID={process.pid}")

        if worker_id < num_workers - 1 and args.launch_interval > 0:
            time.sleep(args.launch_interval)

    failed_workers: list[tuple[int, int]] = []
    for worker_id, process in enumerate(processes):
        process.wait()
        if process.returncode != 0:
            failed_workers.append((worker_id, process.returncode))

    if failed_workers:
        for worker_id, return_code in failed_workers:
            print(f"Worker {worker_id} failed with exit code {return_code}")
        raise RuntimeError("One or more workers failed. Output was not merged.")

    all_results: list[dict[str, Any] | None] = [None] * total

    for worker_id, shard_output in enumerate(shard_output_files):
        if not shard_output.exists():
            raise FileNotFoundError(f"Missing worker output: {shard_output}")

        with open(shard_output, "r", encoding="utf-8") as f:
            shard_results = json.load(f)

        original_indices = list(range(worker_id, total, num_workers))
        if len(shard_results) != len(original_indices):
            raise ValueError(
                f"Worker {worker_id} produced {len(shard_results)} results, "
                f"expected {len(original_indices)}."
            )

        for original_index, result in zip(original_indices, shard_results):
            all_results[original_index] = result

    missing = [index for index, result in enumerate(all_results) if result is None]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} results after merging.")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    if not args.keep_tmp:
        for path in shard_input_files + shard_output_files:
            path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    elapsed = time.time() - start_time

    print("=" * 72)
    print("Generation completed")
    print(f"Total questions: {total}")
    print(f"Output examples: {len(all_results)}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Output file: {args.output_file}")
    print("=" * 72)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate six-dimensional frame-selection weights for video QA."
    )

    parser.add_argument("--model-path", required=True, help="Model name or path.")
    parser.add_argument("--input-file", default="lvb_val.json", help="Input JSON file.")
    parser.add_argument(
        "--output-file",
        default="lvb_val_queries_weight6.json",
        help="Output JSON file.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU ids, for example: 0,1,2.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--tmp-dir", default="tmp_shards")
    parser.add_argument("--launch-interval", type=float, default=2.0)
    parser.add_argument("--append-options", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--shard-file", default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.worker:
        if not args.shard_file or not args.output_file:
            raise ValueError("Worker mode requires --shard-file and --output-file.")
        run_worker(args)
    else:
        run_scheduler(args)


if __name__ == "__main__":
    main()