import requests
from bs4 import BeautifulSoup
from html.parser import HTMLParser

response = requests.get("https://en.wikipedia.org/wiki/Main_Page")

if response.status_code == 200:
        
        soup = BeautifulSoup(response.content, "html.parser")

        paragraph = soup.find(id = "mp-dyk-h2")

        print(paragraph.get_text())

else:
        print("Request Failed")