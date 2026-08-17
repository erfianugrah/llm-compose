"""Register local bench models with bfcl-eval at interpreter startup.

bfcl generate/evaluate resolve --model via MODEL_CONFIG_MAPPING[name] with a
hard KeyError for unknown names, and bfcl-eval 2026.3.23 ships no generic
"local OpenAI endpoint" entry. run-evals.py sets BFCL_LOCAL_MODELS to the
GGUF-stem model id it is about to benchmark; this shim maps each listed id to
the OpenAICompletionsHandler (which reads OPENAI_API_KEY / OPENAI_BASE_URL),
so any OpenAI-compatible local endpoint (llama-server via model_proxy) works.

Installed into the bfcl venv's site-packages by Dockerfile.eval - Python
auto-imports sitecustomize at startup, so both `bfcl generate` and
`bfcl evaluate` (and the eval_checker's own lookups) see the registration.
"""

import os


def _register_local_models():
    names = os.environ.get("BFCL_LOCAL_MODELS", "")
    if not names:
        return
    try:
        from bfcl_eval.constants.model_config import (
            MODEL_CONFIG_MAPPING,
            ModelConfig,
        )
        from bfcl_eval.model_handler.api_inference.openai_completion import (
            OpenAICompletionsHandler,
        )
    except Exception:
        return  # bfcl not installed (e.g. global interpreter) - stay silent
    for name in names.split(","):
        name = name.strip()
        if not name:
            continue
        # bfcl evaluate does model_name.replace("_", "/") for config lookups
        # (their convention: file paths use _ for /). Our GGUF stems contain
        # literal underscores (Q4_K_M), so register BOTH forms.
        for key in {name, name.replace("_", "/")}:
            if key not in MODEL_CONFIG_MAPPING:
                MODEL_CONFIG_MAPPING[key] = ModelConfig(
                    model_name=name,
                    display_name=name,
                    url=None,
                    org="local",
                    license="local",
                    model_handler=OpenAICompletionsHandler,
                    input_price=None,
                    output_price=None,
                    is_fc_model=False,  # prompt mode: generic, no native-FC assumptions
                    underscore_to_dot=False,
                )


_register_local_models()
