"""Sync evaluation logic for worker processing."""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from ...events import TokenGenerated
from ...events.generators import RunsGenerator
from ...llms.huggingface_llm import HuggingFaceLLM
from ...llms.registry import ModelRegistry
from ...schemas.chat import ChatMessage
from .scorers import exact_match, f1_score
from .scorers.common import normalize_answer

logger = logging.getLogger(__name__)


class Evaluator:
    """Main evaluator class for running evaluations."""

    def __init__(self):
        self.scorers = {
            "exact_match": exact_match.score,
            "f1": f1_score.score,
        }

    def run_evaluation(self, eval_id: str, model: str, eval_name: str, queue_job_id: int, **kwargs) -> Dict:
        """Run an evaluation synchronously.

        Args:
            eval_id: Evaluation run ID
            model: Model to evaluate
            eval_name: Name of the evaluation (gsm8k, humaneval, etc.)
            queue_job_id: Queue job ID for status updates
        Returns:
            Dict with evaluation results
        """
        logger.info(f"Starting evaluation {eval_id} for model {model} on {eval_name}")
        logger.info(f"Received kwargs: {list(kwargs.keys())}")
        logger.info(f"Metadata: {kwargs.get('metadata', {})}")
        try:
            metadata = kwargs
            data_source = metadata.get("data_source", {})
            eval_def_id = metadata.get("eval_def_id")
            inline_content = metadata.get("inline_content")
            max_samples = metadata.get("max_samples")
            logger.info(
                f"inline_content present: {inline_content is not None}, length: {len(inline_content) if inline_content else 0}"
            )
            if inline_content:
                dataset_content = inline_content
                is_inline = True
                total_samples = len(dataset_content)
                if max_samples is not None and max_samples < total_samples:
                    dataset_content = dataset_content[:max_samples]
                    total_samples = max_samples
            else:
                is_inline = data_source.get("source", {}).get("type") == "inline"
                if is_inline:
                    dataset_content = data_source.get("source", {}).get("content", [])
                    total_samples = len(dataset_content)
                    if max_samples is not None and max_samples < total_samples:
                        dataset_content = dataset_content[:max_samples]
                        total_samples = max_samples
                else:
                    dataset_content = data_source.get("source", {}).get("content", [])
                    total_samples = len(dataset_content)
                    if max_samples is not None and max_samples < total_samples:
                        dataset_content = dataset_content[:max_samples]
                        total_samples = len(dataset_content)
            model_registry = ModelRegistry()
            model_config = model_registry.get_model_config(model)
            if not model_config:
                raise ValueError(f"Model {model} not found")
            llm = HuggingFaceLLM(model_config)
            results = []
            sample_results = []
            completed_samples = 0
            if is_inline:
                dataset_iter = iter(self._load_inline_dataset(dataset_content))
            else:
                dataset_iter = self._load_dataset_generator(eval_name, max_samples=max_samples)
            for idx, sample in enumerate(dataset_iter):
                start_time = time.time()
                sampling_params = data_source.get("sampling_params")
                output = self._generate_output(llm, sample["input"], sampling_params)
                response_time = time.time() - start_time
                scores = self._score_output(output, sample["expected"], eval_name)
                logger.info(f"Sample {idx}: scores={scores}")
                results.append(scores)
                max_score = max(scores.values()) if scores else 0.0
                passed = max_score >= 0.5
                tokens_used = len(sample["input"].split()) + len(output.split())
                sample_result = {
                    "sample_index": idx,
                    "input": sample["input"],
                    "expected_output": sample["expected"],
                    "actual_output": output,
                    "score": max_score,
                    "passed": passed,
                    "response_time": response_time,
                    "tokens_used": tokens_used,
                    "metadata": {"scores": scores, "eval_name": eval_name},
                }
                sample_results.append(sample_result)
                completed_samples = idx + 1
                if idx % 10 == 0:
                    logger.info(f"Progress: {completed_samples}/{total_samples if total_samples else '?'} samples")
            final_results = self._aggregate_results(results)
            logger.info(f"Evaluation {eval_id} completed with results: {final_results}")
            return {"aggregate": final_results, "samples": sample_results}
        except Exception as e:
            logger.error(f"Evaluation {eval_id} failed: {str(e)}")
            raise

    def _load_dataset_generator(self, eval_name: str, split: str = "test", max_samples: Optional[int] = None):
        """Load evaluation dataset from HuggingFace datasets.

        Args:
            eval_name: Name of the evaluation dataset
            split: Dataset split to use (default: "test")
            max_samples: Maximum number of samples to load (default: all)
        Yields:
            Dicts with 'input' and 'expected' keys
        """
        try:
            dataset_map = {
                "gsm8k": {
                    "name": "gsm8k",
                    "config": "main",
                    "input_field": "question",
                    "expected_field": "answer",
                },
                "humaneval": {
                    "name": "openai_humaneval",
                    "config": None,
                    "input_field": "prompt",
                    "expected_field": "canonical_solution",
                },
                "mmlu": {
                    "name": "cais/mmlu",
                    "config": "all",
                    "input_field": "question",
                    "expected_field": "answer",
                },
            }
            if eval_name not in dataset_map:
                raise ValueError(f"Unknown dataset: {eval_name}. Supported datasets: {list(dataset_map.keys())}")
            config = dataset_map[eval_name]
            dataset = load_dataset(config["name"], config.get("config"), split=split)
            count = 0
            for idx, item in enumerate(dataset):
                if max_samples and idx >= max_samples:
                    break
                input_text, expected = self._format_sample(eval_name, item, config)
                yield {
                    "input": input_text,
                    "expected": expected,
                }
                count += 1
            logger.info(f"Yielded {count} samples from {eval_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {eval_name}: {e}")
            raise ValueError(f"Failed to load dataset {eval_name}: {str(e)}") from e

    def _load_dataset(self, eval_name: str, split: str = "test", max_samples: Optional[int] = None) -> List[Dict]:
        """Load dataset using generator to avoid memory issues."""
        return list(self._load_dataset_generator(eval_name, split, max_samples))

    def _load_inline_dataset(self, content: List[Dict]) -> List[Dict]:
        """Load dataset from inline data source."""
        samples = []
        for item_wrapper in content:
            item = item_wrapper.get("item", {})
            samples.append(
                {
                    "input": item.get("input", ""),
                    "expected": item.get("expected", item.get("ground_truth", "")),
                }
            )
        logger.info(f"Loaded {len(samples)} samples from inline data")
        return samples

    def _format_sample(self, eval_name: str, item: Dict, config: Dict) -> Tuple[str, str]:
        """Format a dataset sample based on eval type.

        Returns:
            Tuple of (formatted_input, expected_output)
        """
        if eval_name == "gsm8k":
            question = item[config["input_field"]]
            answer = item[config["expected_field"]].split("#### ")[-1].strip()
            prompt = f"Question: {question}\n\nProvide the numerical answer only:"
            return prompt, answer
        elif eval_name == "humaneval":
            prompt = item[config["input_field"]] + "\n# Complete the function:"
            solution = item[config["expected_field"]].strip()
            return prompt, solution
        elif eval_name == "mmlu":
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            prompt = f"{question}\n\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65 + i)}. {choice}\n"
            prompt += "\nAnswer with just the letter:"
            answer = chr(65 + answer_idx)
            return prompt, answer
        else:
            return item[config["input_field"]], item[config["expected_field"]]

    def _generate_output(self, llm: HuggingFaceLLM, input_text: str, sampling_params: Optional[Dict] = None) -> str:
        """Generate model output for input.

        Args:
            llm: The language model instance
            input_text: Input prompt text
            sampling_params: Optional sampling parameters (temperature, max_tokens, top_p, seed)
        """
        messages = [ChatMessage(role="user", content=input_text)]
        params = sampling_params or {}
        temperature = params.get("temperature", 1.0)
        max_tokens = params.get("max_completion_tokens", params.get("max_tokens", 2048))
        top_p = params.get("top_p", 1.0)
        seed = params.get("seed", None)
        logger.debug(
            f"Using sampling params: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}, seed={seed}"
        )
        generator = RunsGenerator(llm)

        async def collect_response():
            response_text = ""
            async for event in generator.generate_events(
                messages, temperature=temperature, max_tokens=max_tokens, enable_tools=False
            ):
                if isinstance(event, TokenGenerated):
                    response_text += event.token
            return response_text

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(collect_response())
            return response.strip()
        finally:
            loop.close()

    def _score_output(self, output: str, expected: str, eval_name: str) -> Dict[str, float]:
        """Score model output against expected answer."""
        scores = {}
        exact_score = self.scorers["exact_match"](output, expected)
        scores["exact_match"] = exact_score
        f1_score_val = self.scorers["f1"](output, expected)
        scores["f1"] = f1_score_val
        if eval_name == "gsm8k":
            expected_nums = re.findall(r"-?\d+\.?\d*", expected)
            output_nums = re.findall(r"-?\d+\.?\d*", output)
            if expected_nums and output_nums:
                for exp_num in expected_nums:
                    if exp_num in output_nums:
                        scores["number_match"] = 1.0
                        return scores
            scores["number_match"] = 0.0
        expected_norm = normalize_answer(expected)
        output_norm = normalize_answer(output)
        if expected_norm in output_norm or output_norm in expected_norm:
            scores["substring_match"] = 0.8
        else:
            scores["substring_match"] = 0.0
        return scores

    def _aggregate_results(self, results: List[Dict[str, float]]) -> Dict:
        """Aggregate evaluation results."""
        if not results:
            return {"error": "No results to aggregate"}
        all_metrics = set()
        for result in results:
            all_metrics.update(result.keys())
        metrics = {}
        for metric in all_metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                metrics[metric] = {
                    "mean": sum(values) / len(values),
                    "total": len(values),
                    "passed": sum(1 for v in values if v >= 0.5),
                }
            else:
                metrics[metric] = {"mean": 0.0, "total": 0, "passed": 0}
        return {"metrics": metrics, "total_samples": len(results)}
