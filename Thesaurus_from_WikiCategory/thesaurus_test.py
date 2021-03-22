# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:01:56 2021

@author: bessyhuang
"""

import json
import requests
from lxml import html



### Category (Wikipedia) ###
w = open('thesaurus.txt', 'w', encoding='utf-8')

def subcat(key):
    s = requests.session()
    s.keep_alive = False
    url = 'https://zh.wikipedia.org/wiki/Category:' + key
    wiki = s.get(url)
    tree = html.fromstring(wiki.text)

    print('cat:' + key)
    term.append(key)
    # 子分類
    subcategories = tree.xpath('//div[@class="CategoryTreeItem"]//a/text()')
    # print(subcategories)
    pages = tree.xpath('//div[@id="mw-pages"]//a/text()')
    # print(pages)

    for s in subcategories:
        if s in term:
            pass
        else:
            subcat(s)
            print(s)

    for p in pages:
        if p in term:
            pass
        else:
            term.append(p)
            print(p)

# =============================================================================
### TEST ###
term = []
subcat('茶飲料')

for t in term:
    w.write(t+'\n')
# =============================================================================

### Redirect (Wikipedia) ###
# https://stackoverflow.com/questions/47537644/python-how-to-get-the-page-wikipedia-will-redirect-me-to

KEYWORD = 'vpn'
query = requests.get(r'https://zh.wikipedia.org/w/api.php?action=query&titles={}&&redirects&format=json'.format(KEYWORD))
data = json.loads(query.text)
print(data)

res = requests.get('https://zh.wikipedia.org/wiki/' + KEYWORD)
doc = html.fromstring(res.content)

for t in doc.xpath("//html/head/title"):
    print(t.text)
for t in doc.xpath("//*[@id='firstHeading']"):
    print(t.text)
    
for t in doc.xpath("//link[contains(@rel, 'canonical')]"):
    new_url = str(t.attrib['href'])
    print(new_url)
