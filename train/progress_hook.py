"""
Monkey-patch tqdm to write progress to a JSON file.
Loaded via .pth file in site-packages — runs at Python startup.
Only activates when TRAIN_PROGRESS_FILE env var is set.
"""
import os as _os

if _os.environ.get("TRAIN_PROGRESS_FILE"):
    def _setup_progress_hook():
        import os
        import json
        import time

        progress_file = os.environ["TRAIN_PROGRESS_FILE"]
        last_write = [0.0]
        write_interval = 2

        try:
            from tqdm import tqdm
        except ImportError:
            return

        _orig_update = tqdm.update
        _orig_set_postfix = tqdm.set_postfix

        def _write(self):
            now = time.time()
            if now - last_write[0] < write_interval:
                return
            last_write[0] = now
            try:
                data = {
                    "step": int(self.n),
                    "total": int(self.total) if self.total else 0,
                    "elapsed": round(now - self.start_t, 1) if hasattr(self, "start_t") else 0,
                }
                pd = getattr(self, "_pd", {})
                for key in ("avr_loss", "loss"):
                    if key in pd:
                        data["loss"] = float(pd[key])
                        break
                with open(progress_file, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass

        def _update(self, n=1):
            r = _orig_update(self, n)
            _write(self)
            return r

        def _set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            if not hasattr(self, "_pd"):
                self._pd = {}
            if ordered_dict:
                self._pd.update(ordered_dict)
            self._pd.update(kwargs)
            r = _orig_set_postfix(self, ordered_dict, refresh, **kwargs)
            _write(self)
            return r

        tqdm.update = _update
        tqdm.set_postfix = _set_postfix

    _setup_progress_hook()
    del _setup_progress_hook
