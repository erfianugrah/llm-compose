import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import sys

# Ensure we can import llmc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llmc.bench.context import run_context_sweep, fill_to_tokens

class TestBenchContext(unittest.TestCase):
    def test_fill_to_tokens_exact_size(self):
        # mock tokenize: 1 token per 4 chars
        def mock_tokenize(text: str) -> list[str]:
            return [text[i:i+4] for i in range(0, len(text), 4)]
        
        target = 10
        source = "abcd" * 100
        result = fill_to_tokens(target, source, mock_tokenize)
        self.assertEqual(len(mock_tokenize(result)), target)

    def test_fill_to_tokens_small_source(self):
        def mock_tokenize(text: str) -> list[str]:
            return list(text)
        
        target = 20
        source = "abc" # too small
        result = fill_to_tokens(target, source, mock_tokenize)
        self.assertGreaterEqual(len(mock_tokenize(result)), target)

    @patch("llmc.bench.context.load_all")
    @patch("llmc.bench.context.ProxyClient")
    @patch("llmc.bench.context.store")
    @patch("urllib.request.urlopen")
    @patch("os.symlink")
    @patch("pathlib.Path.unlink")
    @patch("pathlib.Path.write_text") # Wait, typo in my head, it's Path.write_text
    def test_run_context_sweep_dry_run(self, mock_write, mock_unlink, mock_symlink, mock_urlopen, mock_store, mock_client, mock_load_all):
        # Setup
        mock_preset = MagicMock()
        mock_preset.name = "test-preset"
        mock_preset.model.file = "test.gguf"
        mock_preset.model.repo = "test-repo"
        mock_load_all.return_value = {"test-preset": mock_preset}
        
        mock_store.REPO_ROOT = Path("/tmp")
        mock_store.make_record.return_value = MagicMock()
        
        # Execute
        result = run_context_sweep(
            preset_name="test-preset",
            ctx_sizes=[1024],
            slots=1,
            occupancies=[0.5],
            gen_tokens=10,
            dry_run=True
        )
        
        # Verify
        self.assertEqual(result, 0)

    @patch("llmc.bench.context.load_all")
    @patch("llmc.bench.context.ProxyClient")
    @patch("llmc.bench.context.store")
    def test_run_context_sweep_error_preset_not_found(self, mock_store, mock_client, mock_load_all):
        # Setup
        mock_load_all.return_value = {}
        
        # Execute
        result = run_context_sweep(
            preset_name="missing",
            ctx_sizes=[1024],
            slots=1,
            occupancies=[0.5],
            gen_tokens=10,
            dry_run=True
        )
        
        # Verify
        self.assertEqual(result, 1)

if __name__ == "__main__":
    unittest.main()
