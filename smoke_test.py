"""Week 0 Smoke Test — PreViral"""
import sys

tests = {
    "numpy": lambda: __import__("numpy").__version__,
    "scipy": lambda: __import__("scipy").__version__,
    "sklearn": lambda: __import__("sklearn").__version__,
    "cv2": lambda: __import__("cv2").__version__,
    "PIL": lambda: __import__("PIL").__version__,
    "torch": lambda: __import__("torch").__version__,
    "transformers": lambda: __import__("transformers").__version__,
    "sentence_transformers": lambda: __import__("sentence_transformers").__version__,
    "lightgbm": lambda: __import__("lightgbm").__version__,
    "fastapi": lambda: __import__("fastapi").__version__,
    "vaderSentiment": lambda: "OK",
}

ok, fail = 0, 0
for name, fn in tests.items():
    try:
        version = fn()
        print(f"  OK   {name} ({version})")
        ok += 1
    except Exception as e:
        print(f"  FAIL {name}: {str(e)[:90]}")
        fail += 1

print(f"\n--- SMOKE TEST: {ok} OK, {fail} FAILED ---")
if fail == 0:
    print("All dependencies resolved. Ready to start server.")
