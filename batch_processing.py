from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation")
tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")

def analyze_texts(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = logits.softmax(dim=-1)

    id2label = model.config.id2label
    labels = [id2label[idx] for idx in range(probabilities.size(-1))]
    all_label_prob_pairs = []

    for text_idx, probs in enumerate(probabilities):
        label_prob_pairs = list(zip(labels, probs.tolist()))
        label_prob_pairs.sort(key=lambda item: item[1], reverse=True)
        all_label_prob_pairs.append(label_prob_pairs)

    return all_label_prob_pairs, probabilities.mean(dim=0), labels

def assign_account_label(texts, threshold=0.3):
    all_label_prob_pairs, overall_probabilities, labels = analyze_texts(texts)

    overall_prob_dict = {label: prob.item() for label, prob in zip(labels, overall_probabilities)}

    sorted_labels = sorted(overall_prob_dict.items(), key=lambda x: x[1], reverse=True)
    final_label, final_prob = sorted_labels[0]

    return final_label if final_prob > threshold else "OK", sorted_labels

texts = [
    "I love the world!",
    "I hate the world!",
]

final_label, sorted_overall_probs = assign_account_label(texts)
print(f"Final Account Label: {final_label}")
print("Overall Probabilities:")
for label, probability in sorted_overall_probs:
    print(f"Label: {label} - Probability: {probability:.4f}")
