import requests
from bs4 import BeautifulSoup
from html.parser import HTMLParser

response = requests.get("https://www.reddit.com/r/stocks/new/")

if response.status_code == 200:
        
        soup = BeautifulSoup(response.content, "html.parser")

        paragraph = soup.find(id = "t3_1b898d8-post-rtjson-content")

        print(paragraph.get_text())

else:
        print("Request Failed")