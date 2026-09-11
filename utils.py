import time
from google.genai import errors


def call_with_retry(fn, *args, max_retries=5, **kwargs):
    """Wraps a Gemini API call with exponential backoff on 429s."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except errors.ClientError as e:
            last_error = e
            print(f"[Gemini RAW ERROR, attempt {attempt + 1}/{max_retries}]: {e}")
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait = min(2 ** attempt, 30)  # 1, 2, 4, 8, 16, 30s
                print(f"[Gemini] rate-limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Gemini API still rate-limited after {max_retries} retries. "
        f"Last raw error: {last_error}"
    )