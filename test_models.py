import os, time
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

models_to_test = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-lite-latest"]

for model in models_to_test:
    try:
        t0 = time.time()
        r = client.models.generate_content(
            model=model,
            contents="Say hello",
            config=types.GenerateContentConfig(max_output_tokens=64)
        )
        elapsed = time.time() - t0
        txt = (r.text or "")[:60]
        print(f"OK  {model} ({elapsed:.1f}s) -> {txt}")
    except Exception as e:
        print(f"FAIL {model} -> {str(e)[:120]}")
