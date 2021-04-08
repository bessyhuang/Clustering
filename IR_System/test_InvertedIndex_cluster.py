import pymongo_custom_module as PymongoCM
import text_preprocess as tp
import pandas as pd

from collections import Counter, defaultdict
from opencc import OpenCC
import pickle
import wptools


Question_list, Q_WS_list, A_WS_list, Answer_list, new_Cluster_list, AllField_list = PymongoCM.get_mongodb_row("Library", "FAQ")
FAQ_df = pd.DataFrame({"content": Q_WS_list, "new_Cluster": new_Cluster_list, "answer": Answer_list})

FAQ_df["clean_msg"] = FAQ_df.content.apply(tp.text_process)

# ------------------------------------------------------------------------------------------
zipped = zip(Q_WS_list, new_Cluster_list)
Cluster_set = set(new_Cluster_list)

cluster_index = defaultdict(int)
for i in range(-1, len(Cluster_set) - 1):
    cluster_index['Cluster {}'.format(i)] = i + 1
# print(cluster_index)



def SegList_ClusterGroup(zipped):
    Cluster_dict = defaultdict(list)
	
    for row_Q, row_c in zipped:
        if row_c in Cluster_dict:
            Cluster_dict[row_c] += row_Q
        else:
            Cluster_dict[row_c] = row_Q
    # print(Cluster_dict['Cluster 1'])
    return Cluster_dict

SegString_Cluster_dict = SegList_ClusterGroup(zipped)
Q = SegString_Cluster_dict.values()
cluster = SegString_Cluster_dict.keys()
FAQ_cluster_df = pd.DataFrame({'content': Q, 'category': cluster})

### Input Data
FAQ_cluster_df["clean_msg"] = FAQ_cluster_df.content.apply(tp.text_process)

Q_clean_list = FAQ_cluster_df.clean_msg.tolist()
processed_docs = Q_clean_list

# --------------------------------------------------------------------------------------------
# ====== total_tokens 尚未正規化字彙的出現頻率 ======
NO_norm_vocab_freqs = []
no_norm_doc_list = []
for no_norm_doc in Q_WS_list:
    no_norm_doc_list += no_norm_doc
NO_norm_vocab_freqs.append(Counter(no_norm_doc_list))
total_tokens = len(NO_norm_vocab_freqs[0])
#print('尚未正規化字彙的出現頻率: {}'.format(NO_norm_vocab_freqs))
print('尚未 正規化字彙，有 {} 個\n'.format(total_tokens))


# ====== 建立 vocab_freqs (正規化字彙的出現頻率) ======
vocab_freqs = []
norm_doc_str = ""
for norm_doc in processed_docs:
    norm_doc_str += norm_doc + " "
vocab_freqs.append(Counter(norm_doc_str.split()))
total_vocabs = len(vocab_freqs[0])
#print('已正規化字彙的出現頻率: {}'.format(vocab_freqs))
print('已 正規化字彙，有 {} 個\n'.format(total_vocabs))
print('---------------------------\n')


# ====== 建立 vocab (正規化字彙的索引) ======
vocab = defaultdict(int)
for term in norm_doc_str.split():
    if term not in vocab:
        vocab[term] = len(vocab)
#print('已 正規化字彙 ( {} 個) \n的 索引： {}\n'.format(total_vocabs, vocab))

# ====== doc_TF 儲存每個文件分別的字彙及詞頻Counter ======
doc_TF = []
for norm_doc_STRING in processed_docs:
    norm_doc = list(norm_doc_STRING.split())
    
    TF = Counter(norm_doc) # 計算詞頻
    #print(TF) 
    #TF_Output: Counter({'館際合作': 2, '回覆': 2, '時間': 2, '多久': 2}) -> One document
   
    doc_TF.append(TF)
#print(len(doc_TF) == len(set(new_Cluster_list)))
print('---------------------------\n')




# Cluster : N docs
class Docs_in_Cluster:
    def __init__(self, cluster_index, cluster_WS_list, cluster_Category_list):
        self.cluster_index = cluster_index
        
        self.cluster_len = len(self.cluster_index)
        self.cluster_doc_num = [[] for i in range(self.cluster_len)]
        self.cluster_df = pd.DataFrame({'Q_WS': cluster_WS_list, 'Cat': cluster_Category_list})

        sectors = self.cluster_df.groupby('Cat')
        for ClusterN in self.cluster_index.keys():
            Cluster_ID = self.cluster_index[ClusterN]
            # print('====', ClusterN, Cluster_ID, len(list(sectors.get_group(ClusterN).Q_WS)))
            docs_number = len(list(sectors.get_group(ClusterN).Q_WS))
            self.cluster_doc_num[Cluster_ID].append(docs_number)

    def clusterN_docs(self, cluster_name):
        cluster_ID = cluster_index[cluster_name]
        return self.cluster_doc_num[cluster_ID][0]

    def __str__(self, cluster_name):
        cluster_ID = cluster_index[cluster_name]

        return f"The number of dosc in clusterN. ===> {self.cluster_doc_num}\n\n{cluster_name} ===> {self.cluster_doc_num[cluster_ID][0]} 個文件。\n"

dd = Docs_in_Cluster(cluster_index, Q_WS_list, new_Cluster_list)
# print(dd.__str__("Cluster 529"))

# n = dd.clusterN_docs("Cluster 529")
# print(n)




class InvertedIndex:
    def __init__(self, vocab, doc_TF):
        self.vocab = vocab
        self.vocab_len = len(vocab)
        self.doc_len = [0] * len(doc_TF)
        self.doc_TF = [[] for i in range(len(vocab))]
        self.doc_ids = [[] for i in range(len(vocab))]
        self.doc_freqs = [0] * len(vocab)
        self.total_num_docs = 0
        
        #Longest Document Length
        self.max_doc_len = 0
        
        for docid, term_freqs in enumerate(doc_TF):
            #print('~~~', docid, term_freqs)
            #term_freqs_Output: ~~~ 530 Counter({'書': 6, '有': 5, '嗎': 5, '我': 4, '想': 4, '找': 4, '館藏': 4, '圖書館': 4, '這': 4, '本': 4, '請問': 2, '在': 1, '的': 1, '裡': 1})
            
            doc_len = sum(term_freqs.values())
            #print(doc_len)
            #doc_len_Output: 49
            
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len
            
            self.total_num_docs += 1
            for term, freq in term_freqs.items():
                term_ID = vocab[term]
                self.doc_ids[term_ID].append(docid)
                self.doc_TF[term_ID].append(freq)
                
                self.doc_freqs[term_ID] += 1
        
        """
        ### TEST: 各個詞(vpn) 分別出現在多少文件中 ###
        for docid, term_freqs in enumerate(doc_TF):
            if 'vpn' in term_freqs.keys():
                print('\n-->\t\n', docid, term_freqs)
        
        ### TEST: 各個詞(vpn) 分別出現在哪些 文件ID 中 ###
        #print('\n>>>\t', self.doc_ids)
        
        ### TEST: 詞(vpn) 出現在哪些文件裡，其 "文件ID" 為何？ ###
        print('\n>>>\t', self.doc_ids[self.vocab['vpn']] == [1, 6, 7, 16, 33])
        
        ### TEST: 詞(vpn) 出現在每份文件中的 "出現次數" 為何？ ###
        print('\n+++\t', self.term_in_each_doc_TermFreqs('vpn') == [17, 2, 2, 2, 1] == self.doc_TF[117])
        
	### TEST: doc_id = 530 (第 530 群) 的 文件長度 為何？ ###
        print('\n+++\t', self.doc_len[530], '個詞')
        """


    def num_docs(self):
        return self.total_num_docs

    def term_show_in_docids(self, term):
        term_ID = self.vocab[term]
        return self.doc_ids[term_ID]

    def term_in_each_doc_TermFreqs(self, term):
        term_id = self.vocab[term]
        return self.doc_TF[term_id]

    def term_show_in_N_docs(self, term): 
        term_ID = self.vocab[term]
        #print('\n### term = {}\t term_ID = {}\n各個詞 (每個 vocab) 分別出現在 幾份文件 中 = {}\n'.format(term, term_ID, self.doc_freqs))
        #print(self.doc_freqs[term_ID] == len(self.doc_ids[term_ID]))
        return self.doc_freqs[term_ID]
	
# Inverted Index 相關訊息
INV_INDEX = InvertedIndex(vocab, doc_TF)
print("Number of documents (Cluster -1 ~ Cluster 530) = {}".format(INV_INDEX.num_docs()))
print("Number of unique terms = {}".format(INV_INDEX.vocab_len))


# ======== TF-IDF ========
from math import log, sqrt

# 給定一個查詢(String)和一個索引(Class)，回傳k個文件 (先找term索引_確定檢索範圍，再計算doc_tfidf_排序相似度)
def query_tfidf(query, invindex, k=5):
    # scores 儲存了 docID 和他們的 TF-IDF分數
    scores = Counter()
    N = invindex.num_docs() # Cluster -1 ~ Cluster 529 => 531 個

    query_vector = []
    query_term = []
    for term in query:
        if term in vocab:
            term_show_in_N_docs = invindex.term_show_in_N_docs(term)
            query_idf = log(N / term_show_in_N_docs)
            query_vector.append(query_idf)
            query_term.append(term)
        else:
            query_vector.append(0)
            query_term.append(term)

    print(query_vector)
    print(query_term)
    print('=====================================================\n\n')
    for term in query:
        i = 0
        term_show_in_N_docs = invindex.term_show_in_N_docs(term)
        #print('\nTerm ({}) show in {} documents.\n'.format(term, term_show_in_N_docs))
        
        for docid in invindex.term_show_in_docids(term):
            term_in_each_doc_TermFreqs = invindex.term_in_each_doc_TermFreqs(term)[i]     
            term_Maxfreqs_in_doc = max(invindex.term_in_each_doc_TermFreqs(term))

            doc_len = invindex.doc_len[docid] #每個 doc 的長度

            # tfidf_cal = (1 + log(term_in_each_doc_TermFreqs)) * log(N / term_show_in_N_docs) / doc_len
            # tfidf_cal = (1 + log(term_in_each_doc_TermFreqs)) * log(N / term_show_in_N_docs) / sqrt(doc_len)
            tfidf_cal = log(1 + term_in_each_doc_TermFreqs) * log(N / term_show_in_N_docs) / sqrt(doc_len)

            scores[docid] += tfidf_cal
            i += 1
    return scores.most_common(k)
print()



print('=====================================================')
# 問題
Q_list = []
for doc in FAQ_cluster_df.content:
    Q_list.append(''.join(doc))
    
# 回答
A_list = []
sectors = FAQ_df.groupby('new_Cluster')
sectors_len = len(sectors)
for ClusterN in range(-1, sectors_len -1, 1):
    ClusterN_index = list(sectors.get_group('Cluster {}'.format(ClusterN)).index)[0]
    # print(FAQ_df.loc[ClusterN_index].answer)
    A_list.append(FAQ_df.loc[ClusterN_index].answer)



# ----- wikipedia 擴展關鍵詞 ---------------------------------
with open('./food_dict.pkl', 'rb') as fp:
    wiki_food_dict = pickle.load(fp)
fp.close()


subCategory_dict = defaultdict(str)
for key, subcat_items in wiki_food_dict.items():
    for item in subcat_items:
        subCategory_dict[item] = key
# print(subCategory_dict)


wiki_GroupCategory_list = []
for key in wiki_food_dict.keys():
    wiki_GroupCategory_list.append(wiki_food_dict[key] + [key])
# print('wiki_category_list = ', wiki_GroupCategory_list)


total_wiki = []
for i in wiki_GroupCategory_list:
    total_wiki += i
# print('total_wiki = ', total_wiki)


custom_match_dict = {
    '零食':'食物', '飲料':'飲料', 
    '系統':'電腦', '借閱證':'閱覽證', 
    '團討室':'團體 討論室', '互借':'館際 互借', 
    '智慧財產權':'智財權'
    } # wiki_category : FAQ_vocab 
# ------------------------------------------------------------



# ----- 查詢館藏的停用詞擷取 -----------------------------------
cluster529_stopwords = set()
Cluster529 = list(sectors.get_group('Cluster 529').content)[0]
for ws in Cluster529:
    cluster529_stopwords.add(ws)
# print('***', cluster529_stopwords)
# ------------------------------------------------------------
# ----- 查詢Wikipedia的停用詞擷取 ------------------------------
cluster530_stopwords = set()
Cluster530 = list(sectors.get_group('Cluster 530').content)[0]
for ws in Cluster530:
    cluster530_stopwords.add(ws)
# print('***', cluster530_stopwords)
# ------------------------------------------------------------


# 查詢語句
while True:
    query = input("Input:\n")
    # 預處理查詢，為了讓查詢跟索引內容相同
    clean_query1 = query.split() 
    clean_query2 = list(''.join(tp.text_process(clean_query1)).split())
    print(clean_query1, clean_query2)
    print('---------------------------\n')

    # ----- wikipedia 擴展關鍵詞 -------------------------------------
    final_query = []
    for w in clean_query2:
        if (w in total_wiki) and (w not in vocab.keys()):
            # e.g. 品客、零食
            wikiCategory_term = subCategory_dict[w]
            query_term = custom_match_dict[wikiCategory_term]

            final_query.append(query_term)

        elif (w in total_wiki) and (w in vocab.keys()):
            # e.g. 飲料、食物
            final_query.append(w)

        elif w in custom_match_dict.keys():
            query_term = custom_match_dict[w]
            if ' ' in query_term:
                query_list = query_term.split()
                for i in query_list:
                    final_query.append(i)
            else:
                final_query.append(query_term)

        else:
            # 沒有在 FAQ ，也沒有在 wiki            
            final_query.append(w)

    print('Final_query =', final_query)
    print('--------------------------------------------------------')


    results = query_tfidf(final_query, INV_INDEX)
    for rank, res in enumerate(results):

        # ----- 查詢館藏的關鍵字擷取 -----------------------------------
        if res[0] - 1 == 529:
            search_FJULIB_KEYWORD = ""
            for w in final_query: 
                if w not in cluster529_stopwords:
                    search_FJULIB_KEYWORD = w
                    
            print('\n查詢館藏的關鍵字擷取：', search_FJULIB_KEYWORD)
            raw_res = A_list[res[0]]
            final_res = raw_res + search_FJULIB_KEYWORD

        # ----- 查詢Wikipedia的停用詞擷取 ------------------------------
        elif res[0] - 1 == 530:
            search_WIKI_KEYWORD = ""
            for w in final_query: 
                if w not in cluster530_stopwords:
                    search_WIKI_KEYWORD = w
                    
            print('\n查詢 Wikipedia 的關鍵字擷取：', search_WIKI_KEYWORD)
            raw_res = A_list[res[0]]

            # 分類標籤
            # page = wptools.page(search_WIKI_KEYWORD, lang='zh').get_more()
            # c = page.data['categories']
            # final_res = c #raw_res + search_WIKI_KEYWORD

            # 資訊框
            # page = wptools.page(search_WIKI_KEYWORD, lang='zh').get_parse()
            # wikitext = page.data['infobox']
            # for key in page.data['infobox']:
            #     print(key, '\t', page.data['infobox'][key])
            # final_res = wikitext

            # wikidata (資料不齊全)
            # page = wptools.page(search_WIKI_KEYWORD, lang='zh').get_wikidata() #, wikibase='Q51101'
            # wikidata = page.data['wikidata']
            # for key in page.data['wikidata']:
            #     print(key, '\t', page.data['wikidata'][key], '\n')
            # final_res = wikidata

            # 摘要
            page1 = wptools.page(search_WIKI_KEYWORD, lang='zh').get_restbase('/page/summary/')
            summary = page1.data['exrest']
            wikiURL = page1.data['url']

            cc = OpenCC('s2twp')
            final_res = cc.convert(summary) + '\n' + wikiURL

        else:
            final_res = A_list[res[0]]
        # -------------------------------------------------------------

        print("排名 {:2d} DOCID {:8d} ClusterN {:8d} SCORE {:.3f} \n內容 {:}\n回覆 {:}\n".format(rank+1, res[0], res[0]-1, res[1], Q_list[res[0]][:50], final_res))
    print()
    