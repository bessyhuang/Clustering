# Reference: https://www.kaggle.com/jerrykuo7727/word2vec


import pymongo_custom_module as PymongoCM
import pandas as pd

from gensim.models.word2vec import Word2Vec


Question_list, Q_WS_list, Category_list, AllField_list, Answer_list, A_WS_list = PymongoCM.get_mongodb_row("Library", "FAQ")

FAQ_df = pd.DataFrame({"content": Q_WS_list, "category": Category_list, "ans_content": A_WS_list})

# Generate a sample random row or column from the function caller data frame
corpus = FAQ_df.sample(frac=1)
#print(corpus)
"""
DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)

n, frac：設定要抽樣出的筆數，兩者擇一使用。
預設值：frac=None, n=1 (抽出一筆)
* n (int)：要抽出的筆數
* frac (float)：要抽出的比例，0~1
"""

# Train word2vec 
size = 350
model = Word2Vec(corpus.content, size=size, min_count=10, window=1, sg=1, workers=5, iter=5) #min_count=20
print('Vocabulary size:', len(model.wv.vocab))
#print('==>', model.wv.most_similar(positive=["研究"]))

def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, "not found in Word2Vec model!")
    return similar_df

print(most_similar(model, ['研究', '開館', '時間', '借書', '館際合作']))
print()
print(most_similar(model, ['多久', '幾本','網路', '連線', '小間', '逾期']))

# Save model
model.save('word2vec.model')
model = Word2Vec.load('word2vec.model')
#print(model.wv.vocab.keys())



# 視覺化
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['font.size'] = '10'

def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.wv.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]

    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]

    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x, y, word)

    plt.show()

#display_pca_scatterplot(model, ['研究', '開館', '時間', '借書', '館際合作', '多久', '幾本','網路', '連線', '小間', '逾期'])
#display_pca_scatterplot(model, sample=50)





# =======================================================================================
# Word2Vec-based Information Retrieval
from sklearn.metrics.pairwise import cosine_similarity
import re
import text_preprocess as tp

FAQ_df["clean_msg"] = FAQ_df.content.apply(tp.text_process)
FAQ_df['clean_msg_answer'] = FAQ_df.ans_content.apply(tp.text_process)

# Function returning vector reperesentation of a document
def get_embedding_w2v(doc_tokens):
    embeddings = []
    if len(doc_tokens)<1:
        return np.zeros(size)
    else:
        for tok in doc_tokens:
            if tok in model.wv.vocab:
                embeddings.append(model.wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(size))
        # mean the vectors of individual words to get the vector of the document
        return np.mean(embeddings, axis=0)

# Getting Word2Vec Vectors for Testing Corpus and Queries
FAQ_df['vector_Question'] = FAQ_df['clean_msg'].apply(lambda x :get_embedding_w2v(x.split()))
FAQ_df['vector_Answer'] = FAQ_df['clean_msg_answer'].apply(lambda x :get_embedding_w2v(x.split()))




contractions_dict = { "開": "開館", "end":"endnote" }
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def ranking_ir(query):
    # Pre-process Query
    query = query.lower()
    query = expand_contractions(query)
    #print(list("".join(query).split(" ")))
  
    query = tp.text_process(list("".join(query).split(" ")))
    print('===', query)
  
    # generating vector
    vector = get_embedding_w2v(query.split())

    # ranking documents
    documents = FAQ_df[['content', 'category', 'ans_content']].copy()
    documents['similarity'] = FAQ_df['vector_Answer'].apply(lambda x: cosine_similarity(np.array(vector).reshape(1, -1),np.array(x).reshape(1, -1)).item())
    documents.sort_values(by='similarity', ascending=False, inplace=True)

    return documents.head(10).reset_index(drop=True)

print(ranking_ir('請問 圖書館 幾點 開 ？'))
