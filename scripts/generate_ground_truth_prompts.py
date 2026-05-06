import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

from ground_truth.io_utils import load_records, write_jsonl

load_dotenv()

MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview" 
DEFAULT_PLAN_PATH = Path("derived/ground_truth/subsampled_plans.jsonl")
DEFAULT_OUTPUT_PATH = Path("derived/ground_truth/subsampled_prompts.jsonl")
MAX_ROUNDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate iterative prompt rewrites for ground-truth graph descriptions."
    )
    parser.add_argument(
        "--plans-path",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Input JSONL of subsampled plans. Default: %(default)s",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for generated prompt summaries. Default: %(default)s",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=MAX_ROUNDS,
        help="Number of sequential rewrites to generate per graph. Default: %(default)s",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent graph workers for API-bound generation. Default: %(default)s",
    )
    parser.add_argument(
        "--subsample-num",
        type=int,
        default=-1,
        help="How much of the data to randomly sample (for testing). Default: %(default)s",
    )
    return parser.parse_args()


def call_gemini(
    user_prompt: str,
    system_prompt: str = "You are a helpful, concise assistant.",
) -> str:
    """Call Gemini through LiteLLM for one prompt."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "GEMINI_API_KEY is not set. Please check your .env file or environment variables."
        )

    response = completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )
    return response.choices[0].message.content


def summarize_initial(edit_graph: str) -> str:
    prompt = f"""Given the following edit graph and the set of tools, draft a fully complete instruction that a professional audio producer would specify to recreate it. 
    Please return only the instruction and make it sound conversational.

Edit graph: {edit_graph}
"""
    return call_gemini(prompt)


def summarize_round(description: str, round_idx: int) -> str:
    
    prompt = f"""Rewrite the following music production instruction so it becomes slightly more abstract and less technically precise than the previous version.
        
        Description: {description}"""
    return call_gemini(prompt)


def build_prompt_chain(record: dict[str, object], max_rounds: int) -> dict[str, object]:
    graph = str(record["graph_description"])
    descs = [summarize_initial(graph)]
    for round_idx in range(1, max_rounds):
        descs.append(summarize_round(descs[-1], round_idx))
    return {
        "input_audio": str(record["audio_path"]),
        "clip_id": str(record["clip_id"]),
        "plan_id":  str(record["plan_id"]),
        "source": str(record["target_stem"]),
        "prompt_variants": descs,
        "metadata": record
    }


def safe_build_prompt_chain(
    job: tuple[int, dict[str, object]],
    max_rounds: int,
) -> tuple[int, dict[str, object]]:
    index, record = job
    result = build_prompt_chain(record=record, max_rounds=max_rounds)
    return index, result


def main() -> None:
    args = parse_args()
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1.")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")

    data = load_records(args.plans_path)

    if args.subsample_num >0:
        data = random.sample(data,args.subsample_num)

    jobs = [(index, row) for index, row in enumerate(data)]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        ordered_results = sorted(
            executor.map(
                lambda job: safe_build_prompt_chain(
                    job,
                    max_rounds=args.max_rounds,
                ),
                jobs,
            ),
            key=lambda item: item[0],
        )

    write_jsonl(args.output_path, (result for _, result in ordered_results))


if __name__ == "__main__":
    main()
