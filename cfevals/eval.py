from __future__ import annotations

import abc
import concurrent.futures
import hashlib
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from cfevals.record import RecorderBase, default_recorder


@dataclass
class EvalResult:
    sample_id: str
    metrics: dict
    error: str | None = None


class Eval(abc.ABC):
    def __init__(self, *, seed: int = 0, max_samples: int | None = None):
        self.seed = seed
        self.max_samples = max_samples

    @abc.abstractmethod
    def eval_sample(self, sample: dict, *, recorder: RecorderBase) -> EvalResult:
        raise NotImplementedError

    def run(self, samples: Sequence[dict], *, recorder: RecorderBase | None = None) -> List[EvalResult]:
        recorder = recorder or default_recorder()
        return self.eval_all_samples(
            samples,
            self.eval_sample,
            seed=self.seed,
            max_samples=self.max_samples,
            recorder=recorder,
        )

    @staticmethod
    def eval_all_samples(
        samples: Sequence[dict],
        eval_sample_fn,
        *,
        seed: int,
        max_samples: int | None,
        recorder: RecorderBase,
    ) -> List[EvalResult]:
        rng = np.random.default_rng(seed)
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        if max_samples is not None:
            indices = indices[: max_samples]

        use_sequential = os.environ.get("EVALS_SEQUENTIAL") == "1"
        threads_env = os.environ.get("EVALS_THREADS")
        max_workers = int(threads_env) if threads_env else None

        def seed_for_sample(sample_id: str) -> int:
            digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
            return int(digest[:8], 16) ^ seed

        def run_one(idx: int) -> EvalResult:
            sample = samples[idx]
            sample_id = str(sample.get("sample_id", idx))
            sample_seed = seed_for_sample(sample_id)
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            recorder.set_sample_id(sample_id)
            try:
                return eval_sample_fn(sample, recorder=recorder)
            except Exception as exc:  # noqa: BLE001
                recorder.record_event(
                    "forecast_error", {"error": str(exc), "sample_id": sample_id}, sample_id=sample_id
                )
                return EvalResult(sample_id=sample_id, metrics={}, error=str(exc))

        results: List[EvalResult] = []
        if use_sequential or (max_workers == 1):
            for idx in indices:
                results.append(run_one(idx))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, idx) for idx in indices]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return results
