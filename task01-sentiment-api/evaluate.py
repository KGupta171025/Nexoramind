import time
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Ensure app modules are importable
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.model import SentimentClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

# Curated multi-domain evaluation benchmark (positive, negative, neutral)
BENCHMARK_DATASET = [
    # Positive samples
    {"text": "I absolutely love this new update! The performance is astonishing.", "label": "positive"},
    {"text": "Exceptional customer service, resolved my issue within minutes.", "label": "positive"},
    {"text": "The camera quality on this phone is simply phenomenal.", "label": "positive"},
    {"text": "Great value for money, highly recommend it to everyone.", "label": "positive"},
    {"text": "A heartwarming and brilliantly acted film that deserves awards.", "label": "positive"},
    {"text": "Fast shipping, pristine condition, and very easy to set up.", "label": "positive"},
    {"text": "The battery life exceeds all my expectations, lasts 2 full days.", "label": "positive"},
    {"text": "Delicious food and fantastic ambiance, will definitely return!", "label": "positive"},
    {"text": "Incredible sound clarity and very comfortable to wear for hours.", "label": "positive"},
    {"text": "Super intuitive UI and smooth user experience throughout.", "label": "positive"},

    # Negative samples
    {"text": "Terrible build quality, broke within two days of normal use.", "label": "negative"},
    {"text": "Worst experience ever. The app keeps crashing on startup.", "label": "negative"},
    {"text": "Customer support was unhelpful, dismissive, and extremely rude.", "label": "negative"},
    {"text": "Completely disappointed. Overpriced and underdelivered on promises.", "label": "negative"},
    {"text": "The food arrived cold, soggy, and two hours after the estimated time.", "label": "negative"},
    {"text": "Defective product, unusable right out of the box. Requesting refund.", "label": "negative"},
    {"text": "Horrendous battery drain, device heats up uncomfortably within minutes.", "label": "negative"},
    {"text": "Dull plot, terrible pacing, and completely unconvincing acting.", "label": "negative"},
    {"text": "Constant connection drops and unacceptable latency spikes.", "label": "negative"},
    {"text": "Misleading advertising. Does not have the features promised.", "label": "negative"},

    # Neutral samples
    {"text": "The meeting is scheduled for 10:00 AM in Conference Room B.", "label": "neutral"},
    {"text": "The package was shipped via standard ground transit yesterday.", "label": "neutral"},
    {"text": "The company announced its Q3 financial earnings report today.", "label": "neutral"},
    {"text": "The flight departs from Terminal 2 at gate 45.", "label": "neutral"},
    {"text": "The documentation explains how to configure environment variables.", "label": "neutral"},
    {"text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "label": "neutral"},
    {"text": "The store operates Monday through Friday between 9 AM and 6 PM.", "label": "neutral"},
    {"text": "The train arrived at platform 3 on schedule.", "label": "neutral"},
    {"text": "The system restart procedure takes approximately three minutes.", "label": "neutral"},
    {"text": "The file contains a list of email addresses and phone numbers.", "label": "neutral"},
]


def run_evaluation():
    logger.info("Initializing SentimentClassifier for evaluation...")
    classifier = SentimentClassifier()

    texts = [item["text"] for item in BENCHMARK_DATASET]
    true_labels = [item["label"] for item in BENCHMARK_DATASET]

    logger.info(f"Running evaluation across {len(texts)} benchmark samples...")

    # Measure latency per sample
    latencies_ms = []
    predicted_labels = []
    confidences = []

    for text in texts:
        t0 = time.perf_counter()
        pred = classifier.predict([text])[0]
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        predicted_labels.append(pred["sentiment"])
        confidences.append(pred["confidence"])

    # Batch latency test (batch of 10)
    batch_10 = texts[:10]
    t0_b = time.perf_counter()
    _ = classifier.predict(batch_10)
    t1_b = time.perf_counter()
    batch_10_latency_ms = (t1_b - t0_b) * 1000

    # Calculate metrics
    classes = ["positive", "neutral", "negative"]
    acc = accuracy_score(true_labels, predicted_labels)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=classes, average="macro"
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=classes, average="weighted"
    )

    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=classes, average=None
    )

    cm = confusion_matrix(true_labels, predicted_labels, labels=classes).tolist()

    report_str = classification_report(true_labels, predicted_labels, labels=classes)

    metrics = {
        "model_name": classifier.model_name,
        "sample_count": len(texts),
        "overall": {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(prec_weighted), 4),
            "weighted_recall": round(float(rec_weighted), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "mean_confidence": round(float(np.mean(confidences)), 4),
        },
        "per_class": {
            cls: {
                "precision": round(float(per_class_p[i]), 4),
                "recall": round(float(per_class_r[i]), 4),
                "f1_score": round(float(per_class_f1[i]), 4),
                "support": int(per_class_support[i]),
            }
            for i, cls in enumerate(classes)
        },
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm,
        },
        "latency_benchmarks": {
            "single_sample": {
                "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
                "p50_latency_ms": round(float(np.percentile(latencies_ms, 50)), 2),
                "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 2),
                "throughput_req_per_sec": round(1000.0 / float(np.mean(latencies_ms)), 2),
            },
            "batch_sample_size_10": {
                "total_time_ms": round(batch_10_latency_ms, 2),
                "latency_per_item_ms": round(batch_10_latency_ms / 10.0, 2),
                "throughput_items_per_sec": round(10000.0 / batch_10_latency_ms, 2),
            },
        },
    }

    # Save to metrics.json
    metrics_path = BASE_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("              MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Model: {classifier.model_name}")
    print(f"Total Test Samples: {len(texts)}")
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")
    print(f"Mean Confidence:    {np.mean(confidences):.4f}")
    print("\nClassification Report:")
    print(report_str)
    print("Confusion Matrix (Positive, Neutral, Negative):")
    for row in cm:
        print(f"  {row}")
    print("\nLatency Benchmarks:")
    print(f"  Mean Single Request Latency: {np.mean(latencies_ms):.2f} ms")
    print(f"  P95 Single Request Latency:  {np.percentile(latencies_ms, 95):.2f} ms")
    print(f"  Batch (10 items) Total Time: {batch_10_latency_ms:.2f} ms ({batch_10_latency_ms / 10.0:.2f} ms/item)")
    print(f"  Throughput:                  {10000.0 / batch_10_latency_ms:.2f} items/sec")
    print("=" * 60)
    print(f"Detailed metrics exported to: {metrics_path}\n")

    return metrics


if __name__ == "__main__":
    run_evaluation()
