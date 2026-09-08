"""Unit tests for llmc.bench.needle (pure helpers + one live-path mock).

Live proxy / GPU access is NOT exercised here - the smoke run is separate.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmc.bench import needle as N


def tok4(text: str) -> list:
    """Fake tokenizer: 1 token per 4 chars."""
    return [text[i:i + 4] for i in range(0, len(text), 4)]


class TestNeedleString(unittest.TestCase):
    def test_sentence_shape(self):
        s = N.needle_sentence(4096, 0.5, "halcyon")
        self.assertEqual(s, "The secret codeword for level 4096-0.5 is halcyon.")

    def test_question_shape(self):
        q = N.needle_question(16384, 0.95)
        self.assertEqual(
            q, "What is the secret codeword for level 16384-0.95? Reply with the codeword only.")

    def test_level_format(self):
        self.assertEqual(N.needle_level(8192, 0.25), "8192-0.25")

    def test_word_pool(self):
        self.assertEqual(len(N.NEEDLE_WORDS), 8)
        self.assertEqual(len(set(N.NEEDLE_WORDS)), 8)


class TestOffsetAndFillerMath(unittest.TestCase):
    def test_offset_truncates(self):
        self.assertEqual(N.needle_offset(1000, 0.5), 500)
        self.assertEqual(N.needle_offset(1001, 0.5), 500)
        self.assertEqual(N.needle_offset(1000, 0.0), 0)
        self.assertEqual(N.needle_offset(1000, 1.0), 1000)

    def test_filler_target_subtracts_tail(self):
        # target = ctx - gen - 512; filler = target - tail tokens
        ctx, gen = 4096, 64
        tail = len(tok4(N.needle_sentence(ctx, 0.5, "word") + "\n"
                        + N.needle_question(ctx, 0.5)))
        self.assertEqual(N.cell_filler_target(ctx, 0.5, gen, tok4),
                         ctx - gen - N.NEEDLE_HEADROOM - tail)

    def test_filler_target_negative_when_ctx_too_small(self):
        # target = ctx - gen - NEEDLE_HEADROOM; tail is 31 tokens (tok4) at
        # this ctx/depth. Boundary moves with the headroom constant.
        hr = N.NEEDLE_HEADROOM
        # gen=64, tail=31 -> fits iff ctx > 64 + hr + 31
        self.assertLessEqual(N.cell_filler_target(64 + hr + 31, 0.5, 64, tok4), 0)
        self.assertGreater(N.cell_filler_target(64 + hr + 32, 0.5, 64, tok4), 0)


class TestScoreHit(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(N.score_hit("halcyon", "halcyon"))

    def test_case_insensitive(self):
        self.assertTrue(N.score_hit("halcyon", "The codeword is HALCYON."))
        self.assertTrue(N.score_hit("halcyon", "hAlCyOn"))

    def test_word_boundary_blocks_substring(self):
        self.assertFalse(N.score_hit("halcyon", "halcyonite"))
        self.assertFalse(N.score_hit("halcyon", "xhalcyon"))

    def test_missing(self):
        self.assertFalse(N.score_hit("halcyon", "I do not know."))

    def test_punctuation_boundaries(self):
        self.assertTrue(N.score_hit("halcyon", "halcyon."))
        self.assertTrue(N.score_hit("halcyon", "(halcyon)!"))


class TestExpandCells(unittest.TestCase):
    def test_grid_shape_and_order(self):
        cells = N.expand_cells([4096, 8192], [0.25, 0.75], 2)
        self.assertEqual(cells, [
            (4096, 0.25, 0), (4096, 0.25, 1),
            (4096, 0.75, 0), (4096, 0.75, 1),
            (8192, 0.25, 0), (8192, 0.25, 1),
            (8192, 0.75, 0), (8192, 0.75, 1),
        ])

    def test_single_cell(self):
        self.assertEqual(N.expand_cells([4096], [0.5], 1), [(4096, 0.5, 0)])


class TestRunNeedle(unittest.TestCase):
    def _preset(self, name="qwen38-ninfer"):
        p = MagicMock()
        p.name = name
        p.model.file = "qwen38.ninfer"
        p.model.repo = "org/repo"
        p.vram_gb = 22.0
        return p

    def _ctx_patches(self):
        return [
            patch("llmc.bench.context.build_corpus", return_value="abcd" * 100000),
            patch("llmc.bench.context._chat"),
            patch("llmc.bench.context._register_ephemeral"),
            patch("llmc.bench.context._delete_ephemeral"),
        ]

    @patch("llmc.bench.needle.load_all")
    def test_unknown_preset_returns_1(self, mock_load_all):
        mock_load_all.return_value = {}
        rc = N.run_needle("nope", [0.5], [4096], log=lambda *_: None, tokenize_fn=tok4)
        self.assertEqual(rc, 1)

    def test_live_run_all_hit_returns_0_and_records(self):
        from contextlib import ExitStack
        p = self._preset()
        with patch("llmc.bench.needle.load_all", return_value={"qwen38-ninfer": p}), \
             patch("llmc.bench.needle.ProxyClient") as mock_client_cls, \
             patch("llmc.bench.needle.store") as mock_store, \
             ExitStack() as stack:
            (mock_corpus, mock_chat, mock_register, mock_delete) = (stack.enter_context(x) for x in self._ctx_patches())
            client = MagicMock()
            mock_client_cls.return_value = client
            mock_store.make_record.side_effect = lambda kind, preset, path, metrics, rid: {
                "kind": kind, "preset": preset, "metrics": metrics}

            def chat(proxy, model, prompt, max_tokens, timeout):
                if max_tokens == 1:
                    return {}  # warm-up
                word = prompt.split("is ")[1].split(".")[0]
                return {"choices": [{"message": {"content": word}}]}
            mock_chat.side_effect = chat

            logs = []
            rc = N.run_needle("qwen38-ninfer", [0.25, 0.95], [4096], runs=2,
                              gen_tokens=64, log=logs.append, tokenize_fn=tok4)
            self.assertEqual(rc, 0)

            # ephemeral preset registered per ctx + lifecycle bookkeeping
            mock_register.assert_called_once()
            mock_delete.assert_called_once()
            client.set_mode.assert_any_call("llm", model="needle-4096", owner="bench-needle")
            client.set_mode.assert_any_call("llm", model="qwen38-ninfer")  # restore

            # one record per (ctx, depth, run): 1 ctx x 2 depths x 2 runs
            recs = mock_store.make_record.call_args_list
            self.assertEqual(len(recs), 4)
            self.assertTrue(all(c.args[0] == "needle" and c.args[1] == "qwen38-ninfer" for c in recs))
            self.assertEqual(sorted((c.args[3]["depth"], c.args[3]["run"]) for c in recs),
                             [(0.25, 0), (0.25, 1), (0.95, 0), (0.95, 1)])
            self.assertTrue(all(c.args[3]["hit"] == 1 for c in recs))
            mock_store.append.assert_called()

            # 1x2 summary grid on stdout
            joined = "\n".join(logs)
            self.assertIn("0.25", joined)
            self.assertIn("0.95", joined)
            self.assertIn("2/2", joined)

    def test_live_run_all_miss_returns_1(self):
        from contextlib import ExitStack
        with patch("llmc.bench.needle.load_all", return_value={"qwen38-ninfer": self._preset()}), \
             patch("llmc.bench.needle.ProxyClient", return_value=MagicMock()), \
             patch("llmc.bench.needle.store") as mock_store, \
             ExitStack() as stack:
            (mock_corpus, mock_chat, mock_register, mock_delete) = (stack.enter_context(x) for x in self._ctx_patches())
            mock_store.make_record.side_effect = lambda kind, preset, path, metrics, rid: {}
            mock_chat.side_effect = lambda proxy, model, prompt, max_tokens, timeout: (
                {} if max_tokens == 1 else {"choices": [{"message": {"content": "no idea"}}]})

            rc = N.run_needle("qwen38-ninfer", [0.5], [4096], gen_tokens=64,
                              log=lambda *_: None, tokenize_fn=tok4)
            self.assertEqual(rc, 1)

    def _preset_noswap(self, name="qwen38-ninfer"):
        """no-swap path reads the effective_context property, engine, model_id."""
        p = MagicMock()
        p.name = name
        p.engine = "ninfer"
        p.model_id = "qwen3.8-27b-nvfp4"
        # effective_context is a @property on the real Preset - a plain
        # attribute on the mock, NOT .return_value (which makes it callable
        # and hides the property-vs-method confusion that bit the first run).
        type(p).effective_context = property(lambda self: 262144)
        return p

    def test_no_swap_uses_resident_ctx_and_skips_switch(self):
        from contextlib import ExitStack
        # Corpus must cover the resident ctx target: no-swap probes at 262144,
        # so tok4 (1 token per 4 chars) needs a source long enough to fill ~261K.
        big_corpus = "x" * (262144 * 4 + 10000)  # >= target tokens at 4 chars/token
        with patch("llmc.bench.needle.load_all", return_value={"qwen38-ninfer": self._preset_noswap()}), \
             patch("llmc.bench.needle.ProxyClient") as mock_client_cls, \
             patch("llmc.bench.needle.store") as mock_store, \
             ExitStack() as stack:
            (mock_corpus, mock_chat, mock_register, mock_delete) = (stack.enter_context(x) for x in self._ctx_patches())
            mock_corpus.return_value = big_corpus
            mock_store.make_record.side_effect = lambda kind, preset, path, metrics, rid: {}
            mock_chat.side_effect = lambda proxy, model, prompt, max_tokens, timeout: \
                {"choices": [{"message": {"content": "The codeword is brazos."}}]}

            logs = []
            rc = N.run_needle("qwen38-ninfer", [0.5], [4096], gen_tokens=64,
                              log=logs.append, tokenize_fn=tok4, no_swap=True)
            self.assertEqual(rc, 0)
            # no ephemeral registration, no lock, no mode switch
            mock_register.assert_not_called()
            mock_delete.assert_not_called()
            mock_client_cls.return_value.set_lock.assert_not_called()
            mock_client_cls.return_value.set_mode.assert_not_called()
            # ctx overridden to the full resident effective context (262144)
            joined = "\n".join(logs)
            self.assertIn("262144", joined)
            # probe went to the served model id, not an ephemeral needle-<ctx> id
            self.assertTrue(all(c.args[1] == "qwen3.8-27b-nvfp4" for c in mock_chat.call_args_list))


if __name__ == "__main__":
    unittest.main()
