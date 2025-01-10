from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation")
tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")

inputs = tokenizer("hello world", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits

probabilities = logits.softmax(dim=-1).squeeze()

id2label = model.config.id2label
labels = [id2label[idx] for idx in range(len(probabilities))]

label_prob_pairs = list(zip(labels, probabilities))
label_prob_pairs.sort(key=lambda item: item[1], reverse=True)  

for label, probability in label_prob_pairs:
    print(f"Label: {label} - Probability: {probability:.4f}")
