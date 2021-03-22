# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:01:56 2021

@author: bessyhuang
"""

import json
import requests
from lxml import html


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
