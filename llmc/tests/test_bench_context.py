import unittest
import json
import urllib.error
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import sys

# Ensure we can import llmc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmc.bench.context import (
    run_context_sweep,
    fill_to_tokens,
    occupancy_target,
    HEADROOM_TOKENS,
)


def tok4(text: str) -> list:
    """Fake tokenizer: 1 token per 4 chars."""
    return [text[i:i + 4] for i in range(0, len(text), 4)]


class TestFillToTokens(unittest.TestCase):
    def test_exact_size(self):
        source = "abcd" * 500  # 500 tokens
        out = fill_to_tokens(123, source, tok4)
        self.assertEqual(len(tok4(out)), 123)

    def test_zero_target(self):
        self.assertEqual(fill_to_tokens(0, "abcd", tok4), "")

    def test_source_too_small_raises(self):
        with self.assertRaises(ValueError):
            fill_to_tokens(100, "abcd", tok4)  # only 1 token

    def test_uses_full_source_when_exact(self):
        source = "abcd" * 10  # 10 tokens
        out = fill_to_tokens(10, source, tok4)
        self.assertEqual(len(tok4(out)), 10)


class TestOccupancyTarget(unittest.TestCase):
    def test_headroom_math(self):
        # int(ctx*frac) - gen - HEADROOM
        self.assertEqual(occupancy_target(1000, 0.5, 200), 1000 // 2 - 200 - HEADROOM_TOKENS)

    def test_skip_when_no_room(self):
        # ctx too small for any gen budget -> filler target goes non-positive
        self.assertLessEqual(occupancy_target(200, 0.99, 200), 0)
        self.assertLessEqual(occupancy_target(100, 0.5, 200), 0)


class TestRunContextSweep(unittest.TestCase):
    def _preset(self, name="qwen38"):
        p = MagicMock()
        p.name = name
        p.model.file = "Qwen3.8.gguf"
        p.model.repo = "org/repo"
        p.vram_gb = 20.0
        return p

    @patch("llmc.bench.context.load_all")
    def test_dry_run_touches_nothing(self, mock_load_all):
        mock_load_all.return_value = {"qwen38": self._preset()}
        logs = []
        rc = run_context_sweep(
            "qwen38", [196608, 229376], 1, [0.25, 0.98], 200,
            dry_run=True, log=logs.append)
        self.assertEqual(rc, 0)
        joined = "\n".join(logs)
        self.assertIn("ctx=196608", joined)
        self.assertIn("[dry-run]", joined)
        # occupancy 0.98 at 229376 leaves headroom; 0.25 always fine
        self.assertIn("filler=", joined)

    @patch("llmc.bench.context.load_all")
    def test_unknown_preset_returns_1(self, mock_load_all):
        mock_load_all.return_value = {}
        rc = run_context_sweep("nope", [1024], 1, [0.5], 200, dry_run=True, log=lambda *_: None)
        self.assertEqual(rc, 1)

    @patch("llmc.bench.context.load_all")
    @patch("llmc.bench.context.ProxyClient")
    @patch("llmc.bench.context.store")
    @patch("llmc.bench.context._model_dir")
    @patch("llmc.bench.context._chat")
    @patch("llmc.bench.context.build_corpus")
    def test_live_sweep_writes_records_and_cleans_up(
        self, mock_corpus, mock_chat, mock_model_dir, mock_store, mock_client_cls, mock_load_all, tmp_path=None
    ):
        import tempfile
        with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as toml_dir:
            mock_load_all.return_value = {"qwen38": self._preset()}
            mock_corpus.return_value = "abcd" * 10000
            mock_chat.return_value = {"timings": {"predicted_per_second": 42.0},
                                      "usage": {"prompt_tokens": 100, "completion_tokens": 200}}
            mock_model_dir.return_value = Path(model_dir)
            (Path(model_dir) / "Qwen3.8.gguf").write_text("fake-gguf")
            client = MagicMock()
            mock_client_cls.return_value = client
            mock_store.make_record.side_effect = lambda *a, **k: {"_rec": True}

            with patch("llmc.bench.context.MAIN_MODELS_DIR", Path(toml_dir)):
                rc = run_context_sweep(
                    "qwen38", [8192], 1, [0.25], 200,
                    dry_run=False, log=lambda *_: None, tokenize_fn=tok4)

            self.assertEqual(rc, 0)
            # throwaway TOML + symlink cleaned up afterwards
            self.assertEqual(list(Path(toml_dir).glob("ctx-sweep-*.toml")), [])
            self.assertEqual(list(Path(model_dir).glob("ctx-sweep-*.gguf")), [])
            # locked, unlocked, and restored the source preset
            self.assertTrue(client.set_lock.called)
            client.set_mode.assert_any_call("llm", model="qwen38")


class TestMakeTokenizerFallback(unittest.TestCase):
    """make_tokenizer: llama.cpp /tokenize first, local HF tokenizer on 404.

    transformers is not installed on the host, so the HF path is exercised by
    stubbing _hf_tokenizer; the real AutoTokenizer path is covered by the
    eval container, not here.
    """

    def test_proxy_path_used_when_available(self):
        from llmc.bench import context as C
        resp = MagicMock()
        resp.read.return_value = json.dumps({"tokens": [1, 2, 3]}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            tok = C.make_tokenizer("http://proxy", hf_repo="x/y")
            self.assertEqual(tok("hello"), [1, 2, 3])

    def test_404_falls_back_to_hf_repo(self):
        from llmc.bench import context as C
        fake_hf = MagicMock(return_value=[9, 9])
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
                "u", 404, "nf", {}, None)), \
             patch.object(C, "_hf_tokenizer", return_value=fake_hf) as hf:
            tok = C.make_tokenizer("http://proxy", hf_repo="unsloth/Qwen3.8-27B-GGUF")
            self.assertEqual(tok("hello"), [9, 9])
            hf.assert_called_once_with("unsloth/Qwen3.8-27B-GGUF")

    def test_fallback_is_sticky(self):
        """After the first 404 the local tokenizer is reused, no re-request."""
        from llmc.bench import context as C
        fake_hf = MagicMock(return_value=[1])
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
                "u", 404, "nf", {}, None)) as http, \
             patch.object(C, "_hf_tokenizer", return_value=fake_hf):
            tok = C.make_tokenizer("http://proxy", hf_repo="x/y")
            tok("a"); tok("b"); tok("c")
            self.assertEqual(http.call_count, 1)

    def test_no_hf_repo_reraises(self):
        from llmc.bench import context as C
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
                "u", 404, "nf", {}, None)):
            tok = C.make_tokenizer("http://proxy")
            with self.assertRaises(urllib.error.HTTPError):
                tok("x")


if __name__ == "__main__":
    unittest.main()
