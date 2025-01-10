import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

BEARER_TOKEN = "YOUR_BEARER_TOKEN"

model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation")
tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")

def get_user_id(username):
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("data", {}).get("id")
    else:
        print(f"Error fetching user ID: {response.status_code} - {response.text}")
        return None

def get_user_tweets(user_id, max_results=10):
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "max_results": max_results,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error fetching tweets: {response.status_code} - {response.text}")
        return []

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

if __name__ == "__main__":
    username = input("Enter the Twitter username: ")
    user_id = get_user_id(username)

    if user_id:
        print(f"User ID for {username}: {user_id}")
        tweets = get_user_tweets(user_id)

        if tweets:
            texts = [tweet['text'] for tweet in tweets]
            final_label, sorted_overall_probs = assign_account_label(texts)

            print(f"\nFinal Account Label: {final_label}")
            print("Overall Probabilities:")
            for label, probability in sorted_overall_probs:
                print(f"Label: {label} - Probability: {probability:.4f}")

            print("\nIndividual Tweet Analysis:")
            all_label_prob_pairs, _, _ = analyze_texts(texts)
            for i, tweet_probs in enumerate(all_label_prob_pairs):
                print(f"\nTweet {i+1}: {texts[i]}")
                for label, prob in tweet_probs:
                    print(f"Label: {label} - Probability: {prob:.4f}")
        else:
            print(f"No tweets found for {username}.")
    else:
        print(f"Could not fetch user ID for {username}.")
