import requests
BEARER_TOKEN = "YOUR_BEARER_TOKEN"

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

if __name__ == "__main__":
    username = input("Enter the Twitter username: ")
    user_id = get_user_id(username)
    
    if user_id:
        print(f"User ID for {username}: {user_id}")
        tweets = get_user_tweets(user_id)
        
        if tweets:
            print(f"Recent Tweets of {username}:")
            for tweet in tweets:
                print(f"- {tweet['text']}")
        else:
            print(f"No tweets found for {username}.")
