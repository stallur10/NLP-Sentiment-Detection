from bs4 import BeautifulSoup
from requests import get
from fake_useragent import UserAgent

ua = UserAgent()


def lovely_soup(u):
    r = get(u, headers={'User-Agent': ua.chrome})
    return BeautifulSoup(r.text, 'lxml')


url = 'https://finance.yahoo.com/quote/AMD/community'
soup = lovely_soup(url)



# titles = soup.findAll('p', {'class': 'spcv_list-item'})[:20]

# for title in titles:
#     print(title.text)

titles = []
max_results = 5
for tag in soup.find_all('p'):
    if 'spcv_list-item' in tag.get('class', []):
        titles.append(tag)
        if len(titles) >= max_results:
            break