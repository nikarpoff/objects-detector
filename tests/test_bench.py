"""
Latency bench (SVC-10): the JSON of the EXP-2 schema, the statistics of the shared rule.
"""

import json

from pathlib import Path

import pytest

from app.api_v1.schemas import BenchResult
from bench.latency import main, percentile, summarize

CONTRACT = Path(__file__).resolve().parents[1] / "tests" / "contract"


def test_bench_result_has_exp2_keys(capsys, tmp_path):
    out = tmp_path / "result.json"
    assert main(["--backend", "fake", "--n", "5", "--warmup", "1", "--out", str(out)]) == 0

    result = json.loads(capsys.readouterr().out)
    example = json.loads((CONTRACT / "bench.result.json").read_text())
    assert result.keys() == example.keys()
    assert result["hardware"].keys() == example["hardware"].keys()
    BenchResult.model_validate(result)

    assert (result["service"], result["backend"], result["model"], result["device"]) == ("objects-detector", "fake", "fake", "cpu")
    assert (result["n"], result["warmup"]) == (5, 1)
    assert result["hardware"]["gpu"] is None
    assert result["p50_ms"] <= result["p95_ms"]
    assert json.loads(out.read_text()) == result


def test_bench_unknown_backend_fails(capsys):
    assert main(["--backend", "yolo9000", "--n", "1", "--warmup", "0"]) == 1
    assert "Unknown backend 'yolo9000'" in capsys.readouterr().err


def test_bench_statistics_nearest_rank():
    values = [float(value) for value in range(1, 21)]          # 1..20
    assert percentile(values, 0.5) == 10.0                      # x[ceil(10)-1]
    assert percentile(values, 0.95) == 19.0                     # x[ceil(19)-1]
    assert percentile([7.0], 0.95) == 7.0

    summary = summarize([10.0, 20.0, 30.0, 40.0])
    assert summary == {"p50_ms": 20.0, "p95_ms": 40.0, "mean_ms": 25.0, "std_ms": 12.9}
    assert summarize([5.0])["std_ms"] == 0.0


@pytest.mark.parametrize("argv", [["--n", "0"], ["--warmup", "-1"]], ids=["no_runs", "negative_warmup"])
def test_bench_bad_arguments(argv):
    with pytest.raises(SystemExit):
        main(argv)
