"""Shared test setup.

On Windows, onnx's native library fails to load ("DLL initialization routine failed") once pyarrow's is loaded
(tests/test_feedback.py reads Parquet; the ONNX export tests run later). Loading onnx first avoids the clash.
"""
import contextlib

with contextlib.suppress(ImportError):
    import onnx  # noqa: F401
