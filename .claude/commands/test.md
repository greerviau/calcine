Run the calcine test suite and report results.

```bash
uv run pytest tests/ -v --tb=short
```

If tests fail, identify the root cause — don't just show the error. Check whether the failure is in the test itself or the implementation, and suggest a fix.
