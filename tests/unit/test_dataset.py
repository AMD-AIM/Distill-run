from __future__ import annotations

from pathlib import Path

import pytest

from distill_run.dataset import DatasetSpec, resolve
from distill_run.errors import DataError, DatasetFetchError
from distill_run.utils import read_jsonl


def test_local_path_needs_no_download(tmp_path: Path):
    seed = tmp_path / "seed.jsonl"
    seed.write_text('{"instruction": "hi"}\n', encoding="utf-8")
    result = resolve(DatasetSpec(uri=str(seed)), tmp_path / "cache")
    assert result.path == seed
    assert result.from_cache is False


def test_missing_local_path_fails_before_any_engine_work(tmp_path: Path):
    with pytest.raises(DatasetFetchError, match="does not exist"):
        resolve(DatasetSpec(uri=str(tmp_path / "nope.jsonl")), tmp_path / "cache")


def test_unknown_scheme_is_rejected(tmp_path: Path):
    with pytest.raises(DatasetFetchError, match="unsupported dataset URI scheme"):
        resolve(DatasetSpec(uri="ftp://host/data.jsonl"), tmp_path / "cache")


def test_http_download_then_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = '{"instruction": "from network"}\n'
    calls = {"n": 0}

    class FakeResponse:
        headers = {"Content-Length": str(len(payload))}

        def raise_for_status(self):
            return None

        def iter_content(self, _chunk):
            yield payload.encode()

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class FakeSession:
        def get(self, url, **_kwargs):
            calls["n"] += 1
            assert url == "https://example.com/seed.jsonl"
            return FakeResponse()

    monkeypatch.setattr("distill_run.fetchers.http_session", lambda: FakeSession())

    cache = tmp_path / "cache"
    spec = DatasetSpec(uri="https://example.com/seed.jsonl")

    first = resolve(spec, cache)
    assert first.from_cache is False
    assert first.path.read_text(encoding="utf-8") == payload

    second = resolve(spec, cache)
    assert second.from_cache is True
    assert second.path == first.path
    assert calls["n"] == 1


def test_failed_download_leaves_no_cache_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class Boom:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("network down")

    monkeypatch.setattr("distill_run.fetchers.http_session", lambda: Boom())
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="network down"):
        resolve(DatasetSpec(uri="https://example.com/seed.jsonl"), cache)
    assert list(cache.glob("*.done")) == []
    assert list(cache.glob("*.tmp-*")) == []


def test_cache_key_separates_revisions():
    assert DatasetSpec(uri="hf://org/name").cache_key() != (
        DatasetSpec(uri="hf://org/name", revision="v2").cache_key()
    )


def test_hf_uri_without_the_sdk_reports_a_useful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(DatasetFetchError, match="huggingface_hub"):
        resolve(DatasetSpec(uri="hf://org/name"), tmp_path / "cache")


def test_json_instruction_output_is_normalized_for_swift(tmp_path: Path):
    source = tmp_path / "data.json"
    source.write_text(
        '[{"instruction":"Translate","input":"hello","output":"你好"}]',
        encoding="utf-8",
    )
    result = resolve(DatasetSpec(uri=str(source), purpose="sft"), tmp_path / "cache")
    assert read_jsonl(result.path) == [
        {
            "messages": [
                {"role": "user", "content": "Translate\n\nhello"},
                {"role": "assistant", "content": "你好"},
            ]
        }
    ]


def test_conversations_roles_are_normalized_for_swift(tmp_path: Path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        '{"conversations":[{"from":"human","value":"2+2?"},{"from":"gpt","value":"4"}]}\n',
        encoding="utf-8",
    )
    result = resolve(DatasetSpec(uri=str(source), purpose="sft"), tmp_path / "cache")
    assert read_jsonl(result.path)[0]["messages"] == [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_prompt_normalization_removes_reference_answer(tmp_path: Path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        '{"messages":[{"role":"system","content":"Be concise"},'
        '{"role":"user","content":"2+2?"},{"role":"assistant","content":"4"}]}\n',
        encoding="utf-8",
    )
    result = resolve(DatasetSpec(uri=str(source), purpose="prompt"), tmp_path / "cache")
    assert read_jsonl(result.path)[0]["messages"] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "2+2?"},
    ]


def test_max_samples_is_deterministic(tmp_path: Path):
    source = tmp_path / "data.jsonl"
    source.write_text(
        "".join(f'{{"instruction":"question {index}"}}\n' for index in range(20)),
        encoding="utf-8",
    )
    spec = DatasetSpec(
        uri=str(source),
        purpose="seed",
        max_samples=5,
        shuffle_seed=7,
    )
    first = read_jsonl(resolve(spec, tmp_path / "cache-a").path)
    second = read_jsonl(resolve(spec, tmp_path / "cache-b").path)
    assert len(first) == 5
    assert first == second


def test_directory_prefers_requested_split(tmp_path: Path):
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "train.jsonl").write_text('{"instruction":"train"}\n', encoding="utf-8")
    (source / "test.jsonl").write_text('{"instruction":"test"}\n', encoding="utf-8")
    spec = DatasetSpec(uri=str(source), purpose="seed", split="test")
    result = resolve(spec, tmp_path / "cache")
    assert read_jsonl(result.path) == [{"instruction": "test"}]


def test_sft_rows_without_answers_are_rejected(tmp_path: Path):
    source = tmp_path / "data.jsonl"
    source.write_text('{"instruction":"no answer"}\n', encoding="utf-8")
    with pytest.raises(DataError, match="no rows can be normalized"):
        resolve(DatasetSpec(uri=str(source), purpose="sft"), tmp_path / "cache")
