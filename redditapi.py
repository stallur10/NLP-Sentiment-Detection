import requests

# Replace these with your Reddit API credentials
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
user_agent = 'YOUR_APP_NAME'

# Subreddit you want to query
subreddit = 'subreddit_name'  # Example: 'python'

# Setting up the authentication and headers
auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
data = {
    'grant_type': 'password',
    'username': 'YOUR_REDDIT_USERNAME',
    'password': 'YOUR_REDDIT_PASSWORD'
}
headers = {'User-Agent': user_agent}

# Getting the access token
res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)
token = res.json()['access_token']

# Updating headers with the obtained token
headers = {**headers, 'Authorization': f"bearer {token}"}

# Making a request to get the top 10 posts of the subreddit
res = requests.get(f"https://oauth.reddit.com/r/{subreddit}/top?limit=10",
                   headers=headers)

# Printing the titles of the top 10 posts
for post in res.json()['data']['children']:
    print(post['data']['title'])
