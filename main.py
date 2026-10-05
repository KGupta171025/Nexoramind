import utils.session
import importlib.util
import sys
from pathlib import Path

# Load task01 application cleanly via importlib to avoid root main.py circular reference
TASK01_MAIN = Path(__file__).resolve().parent / "task01-sentiment-api" / "main.py"

spec = importlib.util.spec_from_file_location("task01_sentiment_api", str(TASK01_MAIN))
task01_module = importlib.util.module_from_spec(spec)
sys.modules["task01_sentiment_api"] = task01_module
spec.loader.exec_module(task01_module)

# Expose task01 app at root level
app = task01_module.app

if __name__ == "__main__":
    import uvicorn

    print("Launching Nexoramind Task 01: Sentiment Analysis API on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
