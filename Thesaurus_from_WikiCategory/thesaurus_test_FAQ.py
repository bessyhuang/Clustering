# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:08:59 2021

@author: bessyhuang
"""
from collections import defaultdict
from pymongo import MongoClient
from lxml import html

import requests
import pandas
import json

def get_mongodb_row(DatabaseName, CollectionName):
    client = MongoClient("localhost", 27017)
    db = client[DatabaseName]
    collection = db[CollectionName]

    cursor = collection.find({}, {"_id":1, "LibraryName":1, "ID":1, 
         "Question":1, "Answer":1, "Category":1, "Keyword":1, "RelatedQ":1, 
         "Q_WS":1, "A_WS":1, "QA_WS":1, 
         "Q_POS":1, "A_POS":1, "QA_POS":1, 
         "Q_NER":1, "A_NER":1, "QA_NER":1, 
         "Q_WS | POS":1, "A_WS | POS":1, "QA_WS | POS":1, 
         "adjusted_Category":1})

    Question_list = []
    Q_WS_list = []
    A_WS_list = []
    QA_WS_list = []
    AllField_list = []
    
    for row in cursor:
        Question_list.append(row['Question'])
        Q_WS_list.append(row['Q_WS'])
        A_WS_list.append(row['A_WS'])
        QA_WS_list.append(row['QA_WS'])
        AllField_list.append((row['Question'], row['Q_WS'], row['A_WS'], row['QA_WS'], row['_id']))

    return Question_list, Q_WS_list, A_WS_list, QA_WS_list, AllField_list

Question_list, Q_WS_list, A_WS_list, QA_WS_list, AllField_list = get_mongodb_row('Library', 'FAQ')
WS_dict = defaultdict(int)

for doc in QA_WS_list:
    for ws in doc:
        WS_dict[ws] += 1
# print(WS_dict)


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
    print('subcategories:', subcategories)
    pages = tree.xpath('//div[@id="mw-pages"]//a/text()')
    print('pages', pages)

    for s in subcategories:
        if s in term:
            pass
        else:
            pass
            # subcat(s)
            print('sub =>', s)

    for p in pages:
        if p in term:
            pass
        else:
            term.append(p)
            print('page =>', p)

# =============================================================================
### TEST ###
# term = []
# subcat('茶飲料')
# =============================================================================
from nltk.corpus import stopwords
import string

with open("./StopWords/stop_words.txt", encoding="utf-8") as f:
    stop_words = f.read().splitlines()

CustomStopwords = ['「', '」', '，', '。', '？', '、', '【', '】', '（', '）', '.', '﹖'] + list(string.punctuation)
CustomStopwords += stop_words
STOPWORDS = stopwords.words('english') + CustomStopwords
print('STOPWORDS:', STOPWORDS)

for ws in WS_dict:
    if ws in STOPWORDS:
        pass
    elif '/' in ws:
        pass
    elif '*' in ws:
        pass
    elif '>' in ws:
        pass
    elif '<' in ws:
        pass
    elif '?' in ws:
        pass
    elif ':' in ws:
        pass
    elif '"' in ws:
        pass
    else:
        term = []
        subcat(ws)
        if term == [ws]:
            pass        
        else:
            f = open('./Thesaurus_for_FAQws/thesaurus_{}.txt'.format(ws), 'w', encoding='utf-8')
            for t in term:
                f.write(t + '\n')
