import pymongo_custom_module as PymongoCM
import text_preprocess as tp
import pandas as pd

from collections import Counter, defaultdict

Question_list, Q_WS_list, A_WS_list, Category_list, new_Cluster_list, AllField_list = PymongoCM.get_mongodb_row("Library", "FAQ")
FAQ_df = pd.DataFrame({"content": Q_WS_list, "new_Cluster": new_Cluster_list})

FAQ_df["clean_msg"] = FAQ_df.content.apply(tp.text_process)

# ------------------------------------------------------------------------------------------
zipped = zip(Q_WS_list, new_Cluster_list)
Cluster_set = set(new_Cluster_list)

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
            doc_len = sum(term_freqs.values())
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len
            self.total_num_docs += 1
            for term, freq in term_freqs.items():
                term_id = vocab[term]
                self.doc_ids[term_id].append(docid)
                self.doc_TF[term_id].append(freq)
                self.doc_freqs[term_id] += 1
        
        """
        ### TEST: 各個詞(vpn) 分別出現在多少文件中 ###
        for docid, term_freqs in enumerate(doc_TF):
            if 'vpn' in term_freqs.keys():
                print('\n-->\t\n', docid, term_freqs)
        
        ### TEST: 各個詞(vpn) 分別出現在哪些 文件ID 中 ###
        #print('\n>>>\t', self.doc_ids)
                
        ### TEST: 詞(vpn) 出現在哪些文件裡，其 文件ID 為何？ ###
        print('\n>>>\t', self.doc_ids[self.vocab['vpn']])
        
        ### TEST: 詞(vpn) 出現在每份文件中的 出現次數 為何？ ###
        print('\n+++\t', self.term_in_each_doc_TermFreqs('vpn'))
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
        #print('### term = {}\t term_ID = {}\n各個詞 (每個 vocab) 分別出現在多少文件中：\n{}'.format(term, term_ID, self.doc_freqs))
        return self.doc_freqs[term_ID]
	
# Inverted Index 相關訊息
INV_INDEX = InvertedIndex(vocab, doc_TF)
print("Number of documents (Cluster -1 ~ Cluster 529) = {}".format(INV_INDEX.num_docs()))
print("Number of unique terms = {}".format(INV_INDEX.vocab_len))


# ======== TF-IDF ========
from math import log, sqrt

# 給定一個查詢(String)和一個索引(Class)，回傳k個文件 (先找term索引_確定檢索範圍，再計算doc_tfidf_排序相似度)
def query_tfidf(query, invindex, k=10):
    # scores 儲存了 docID 和他們的 TF-IDF分數
    scores = Counter()
    N = invindex.num_docs() # Cluster -1 ~ Cluster 529 => 531 個
    
    for term in query:
        i = 0
        term_show_in_N_docs = invindex.term_show_in_N_docs(term)
        #print('\nTerm ({}) show in {} documents.\n'.format(term, term_show_in_N_docs))
        
        for docid in invindex.term_show_in_docids(term):
            term_in_each_doc_TermFreqs = invindex.term_in_each_doc_TermFreqs(term)[i]     
            
            doc_len = invindex.doc_len[docid] #每個 doc 的長度
            tfidf_cal = log(1 + term_in_each_doc_TermFreqs) * log(N / term_show_in_N_docs) / sqrt(doc_len)
            scores[docid] += tfidf_cal
            i += 1
    return scores.most_common(k)
print()


# 查詢語句
Q_list = []
for doc in FAQ_cluster_df.content:
	Q_list.append(''.join(doc))
while True:
	query = input("Input:\n")
	# 預處理查詢，為了讓查詢跟索引內容相同
	stemmed_query = query.split()
	results = query_tfidf(stemmed_query, INV_INDEX)
	for rank, res in enumerate(results):
		# e.g 排名 1 DOCID 176 SCORE 0.426 內容 South Korea rose 1% in February from a year earlier, the
		print("排名 {:2d} DOCID {:8d} ClusterN {:8d} SCORE {:.3f} \n內容 {:}".format(rank+1, res[0], res[0]-1, res[1], Q_list[res[0]][:50]))
	print()

