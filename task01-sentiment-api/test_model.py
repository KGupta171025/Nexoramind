from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)


texts = [
    "I love this product. It works perfectly!",
    "This product is terrible and disappointing.",
    "The package arrived on Tuesday.",
]

result = classifier(texts)

for text, scores in zip(texts, result):
    best = max(scores, key=lambda item:item["score"])

    print(f"\nText: {text}")
    print(f"Sentiment: {best['label']}")
    print(f"Confidence: {best['score']:.4f}")
    print("Per-class scores:")


    for item in scores:
        print(f"  {item['label']}: {item['score']:.4f}")