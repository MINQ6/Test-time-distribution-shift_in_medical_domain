def run_inference(*args, **kwargs):
    from utils.inference.pipeline import run_inference as _run_inference

    return _run_inference(*args, **kwargs)

__all__ = ["run_inference"]
