import os

from cfevals.eval import Eval, EvalResult
from cfevals.record import NullRecorder


class DummyEval(Eval):
    def eval_sample(self, sample: dict, *, recorder):
        return EvalResult(sample_id=str(sample["sample_id"]), metrics={"value": sample["value"]})


def test_seed_determinism():
    samples = [{"sample_id": str(i), "value": i} for i in range(10)]
    eval_obj = DummyEval(seed=123)
    os.environ["EVALS_SEQUENTIAL"] = "1"
    results_first = eval_obj.run(samples, recorder=NullRecorder())
    results_second = eval_obj.run(samples, recorder=NullRecorder())
    order_first = [res.sample_id for res in results_first]
    order_second = [res.sample_id for res in results_second]
    assert order_first == order_second
