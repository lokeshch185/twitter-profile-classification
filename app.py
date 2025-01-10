import streamlit as st
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import plotly.express as px

TWITTER_API_BEARER_TOKEN = "YOUR_TWITTER_API_BEARER_TOKEN"

model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation")
tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")

def fetch_twitter_user_id_by_twitter_username(twitter_username):
    url = f"https://api.twitter.com/2/users/by/username/{twitter_username}"
    headers = {"Authorization": f"Bearer {TWITTER_API_BEARER_TOKEN}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("data", {}).get("id")
    else:
        return None

def fetch_user_recent_tweets(twitter_user_id, max_results=10):
    url = f"https://api.twitter.com/2/users/{twitter_user_id}/tweets"
    headers = {"Authorization": f"Bearer {TWITTER_API_BEARER_TOKEN}"}
    params = {"max_results": max_results}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        return []

def analyze_text_moderation(tweet_texts):
    inputs = tokenizer(tweet_texts, return_tensors="pt", padding=True, truncation=True)
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

def determine_account_label(tweet_texts, threshold=0.3):
    all_label_prob_pairs, overall_probabilities, labels = analyze_text_moderation(tweet_texts)
    overall_prob_dict = {label: prob.item() for label, prob in zip(labels, overall_probabilities)}
    sorted_labels = sorted(overall_prob_dict.items(), key=lambda x: x[1], reverse=True)
    determined_label, final_prob = sorted_labels[0]
    return determined_label if final_prob > threshold else "OK", sorted_labels

st.title("Twitter Moderation Analysis")

twitter_username = st.text_input("Enter Twitter Username")
if st.button("Analyze"):
    with st.spinner("Fetching tweets and analyzing..."):
        twitter_user_id = fetch_twitter_user_id_by_twitter_username(twitter_username)
        if twitter_user_id:
            tweets = fetch_user_recent_tweets(twitter_user_id)
            if tweets:
                tweet_texts = [tweet['text'] for tweet in tweets]
                determined_label, sorted_label_probabilities = determine_account_label(tweet_texts)
                all_label_prob_pairs, _, _ = analyze_text_moderation(tweet_texts)

                LABEL_FULL_FORMS = {
                    "OK": "Safe",
                    "H": "Hate Speech",
                    "SH": "Self-Harm",
                    "V": "Violence",
                    "HR": "Haresment",
                    "S": "Sexual Content",
                    "V2": "Violent Content",
                    "S3": "Sexual/Minor",
                    "H2": "Hate/Threatening",
                }

                def get_label_full_form(label):
                    return LABEL_FULL_FORMS.get(label, label)

                overall_chart = px.bar(
                    x=[get_label_full_form(label) for label, _ in sorted_label_probabilities],
                    y=[prob for _, prob in sorted_label_probabilities],
                    labels={"x": "Categories", "y": "Probabilities"},
                    title="Overall Analysis"
                )
                st.plotly_chart(overall_chart)

                st.header("Individual Tweet Analysis")
                for i, tweet_probs in enumerate(all_label_prob_pairs):
                    st.subheader(f"Tweet {i+1}: {tweet_texts[i]}")
                    chart = px.bar(
                        x=[get_label_full_form(label) for label, _ in tweet_probs],
                        y=[prob for _, prob in tweet_probs],
                        labels={"x": "Categories", "y": "Probabilities"},
                        title=f"Tweet {i+1} Analysis"
                    )
                    st.plotly_chart(chart)

                
                st.success(f"Final Account Label: {determined_label}")
            else:
                st.error("No tweets found for this username.")
        else:
            st.error("Could not fetch user ID for this username.")
