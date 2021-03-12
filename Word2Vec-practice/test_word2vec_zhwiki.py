# 1. 將 XML 去除標籤輸出成 TXT 純文字檔
"""
import gensim
import re

# 定義函數：去除非中文的文字與符號（只保留中文字）
def remove_punctuation(line):
    rule = re.compile(r'[^\u4e00-\u9fa5|\s]')
    line = rule.sub('', line)
    return line

# 定義函數：去除多餘的空白符號（只留下一個空白當作間隔）
def remove_redundant_space(line):
    line = re.sub(' +', ' ', line)
    return line

# 下載的 wiki 語料檔
wiki_file = './zhwiki/zhwiki-latest-pages-articles.xml.bz2'

with open('./zhwiki/wiki_texts.txt', 'w', encoding='utf8') as fp:
    wiki = gensim.corpora.WikiCorpus(wiki_file, lemmatize=False, dictionary={})
    # 取出文字部分（原本是 XML 格式，包含很多標籤）
    for text in wiki.get_texts():
        # print(text)
        # text 是一篇文章，表示成字串串列（List）
        s = ' '.join(text)
        t = remove_punctuation(s)
        u = remove_redundant_space(t)
        # 每篇文章一個換行作為間隔，寫入輸出檔案
        fp.write(u + '\n')
fp.close()
"""

# 2. 利用 OpenCC 將 TXT 純文字內容都轉換為繁體
"""
(Method 1)
在 Linux 平台安裝 opencc：apt install opencc（在 Linux 處理速度快很多）
執行：opencc -i ./zhwiki/wiki_texts.txt -o ./zhwiki/wiki_zh_tw.txt -c s2tw.json
最後，輸出成繁體的 wiki_zh_tw.txt 文字檔

(Method 2)
安裝 python 的 opencc 繁簡轉換工具模組
安裝：pip3 install opencc-python-reimplemented
"""

# 3. 運用 python 的 opencc 工具模組執行繁體轉換
"""
from opencc import OpenCC

openCC = OpenCC('s2twp')
#s2twp : 簡體中文 -> 繁體中文 (台灣, 包含慣用詞轉換)
#s2t   : 簡體中文 -> 繁體中文
#s2tw  : 簡體中文 -> 繁體中文 (台灣)

with open('C:/Users/bessyhuang/Downloads/python/wiki/wiki_texts.txt', 'r', encoding='utf-8') as fp:
    s = fp.read()
fp.close()

t = openCC.convert(s)

with open('C:/Users/bessyhuang/Downloads/python/wiki/wiki_zh_tw.txt', 'w', encoding='utf-8') as fp:
    fp.write(t)
fp.close()

print('Conversion Complete!')
"""

# 4. Ckiptagger 斷詞
"""
from ckiptagger import construct_dictionary, WS
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def WordSegment_and_write2file(zhwiki_list):
    ws = WS("./zhwiki/data", disable_cuda=False)

    # (Optional) Create dictionary: wikipedia
    with open('./zhwiki/WikiDict_plus_QAkeywordsDict.pkl', 'rb') as fp:
        WikiDict_plus_QAkeywordsDict = pickle.load(fp)
    fp.close()

    dictionary1 = construct_dictionary(WikiDict_plus_QAkeywordsDict)

    word_sentence_list = ws(zhwiki_list,
        segment_delimiter_set = {":" ,"：" ,":" ,".." ,"，" ,"," ,"-" ,"─" ,"－" ,"──" ,"." ,"……" ,"…" ,"..." ,"!" ,"！" ,"〕" ,"」" ,"】" ,"》" ,"【" ,"）" ,"｛" ,"｝" ,"“" ,"(" ,"「" ,"]" ,")" ,"（" ,"《" ,"[" ,"『" ,"』" ,"〔" ,"、" ,"．" ,"。" ,"." ,"‧" ,"﹖" ,"？" ,"?" ,"?" ,"；" ," 　" ,"" ,"　" ,"" ,"ㄟ" ," :" ,"？" ,"〞" ,"]" ,"／" ,"=" ,"？" ," -" ,"@" ,"." ,"～" ," ：" ,"：" ,"<", ">" ," - " ,"──" ,"~~" ,"`" ,": " ,"#" ,"/" ,"〝" ,"：" ,"'" ,"$C" ,"?" ,"?" ,"*" ,"／" ,"[" ,"." ,"?" ,"-" ,"～～" ,"\""},
        recommend_dictionary = dictionary1, # 效果最好！
        coerce_dictionary = construct_dictionary({'OPAC':2, 'OK':2, '滯還金':2, '浮水印':2, '索書號':2, '圖書館':2}), # 強制字典
    )

    with open('./zhwiki/zhwiki_WS.pkl', 'wb') as fp:
        pickle.dump(word_sentence_list, fp)
    fp.close()
    
    with open('./zhwiki/zhwiki_WS.text', 'w') as f:
        f.write(word_sentence_list)
    f.close()
    
    return word_sentence_list


with open('./zhwiki/wiki_zh_tw.txt', 'r', encoding='utf-8') as f:
    wiki_zh_tw_list = f.readlines()
print(wiki_zh_tw_list[:5])

zhwiki_WS_list = WordSegment_and_write2file(wiki_zh_tw_list)
print("======================= WS finished! ==============================")


import jieba
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    # jieba custom setting.我將結巴的字典換為繁體
    # jieba.set_dictionary('data/dict_big_utf-8.txt')
    # 支持繁体分词更好的词典文件
    # jieba.set_dictionary('C:/Users/bessyhuang/Downloads/python/dict.txt.big')
    # 載入額外的使用者辭典
    jieba.load_userdict('./zhwiki/user_dict.txt')
    
# 沒任何想法，暫時不考慮任何 stopwords
#     # load stopwords set去除停用詞
#     stopwordset = set()
#     with open('data/stop_words_utf-8.txt','r',encoding='utf-8') as sw:
#         for line in sw:
#             stopwordset.add(line.strip('\n'))
            
    ofile = open('./zhwiki/zhwiki_WS.txt', 'w', encoding='utf-8')

    n = 0
    
    with open('./zhwiki/wiki_zh_tw.txt' ,'r', encoding='utf-8') as ifile:
        for line in ifile:
            words = jieba.cut(line, cut_all=False)
            for w in words:
                ofile.write(w + ' ')
                n = n + 1
                if (n % 1000000 == 0):
                    # print('已完成前 %d 行的斷詞' % n)
                    logging.info('已完成前 %d 行的斷詞' % n)

    ifile.close()
    ofile.close()

if __name__ == '__main__':
    main()

print("======================= WS finished! ==============================")

"""
# 5. zhwiki + FAQ Train word2vec
import pymongo_custom_module as PymongoCM
import pandas as pd

from gensim.models.word2vec import Word2Vec

with open('./zhwiki/zhwiki_WS.txt', 'r', encoding='utf-8') as f:
    zhwiki_WS_list = f.readlines()
print(zhwiki_WS_list)

Question_list, Q_WS_list, Category_list, AllField_list, Answer_list, A_WS_list = PymongoCM.get_mongodb_row("Library", "FAQ")

zhwiki_plus_Q_list = zhwiki_WS_list + Q_WS_list
ZQ_df = pd.DataFrame({"content": zhwiki_plus_Q_list})

# Generate a sample random row or column from the function caller data frame
corpus = ZQ_df.sample(frac=1)
print(corpus)

# Train word2vec 
size = 1000
model = Word2Vec(corpus.content, size=size, min_count=10, window=5, sg=1, workers=1000, iter=5) #min_count=20
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

print(most_similar(model, ['covid-19', '開館', '時間', '借書', '館際合作']))
print()
print(most_similar(model, ['多久', '幾本','網路', '連線', '小間', '逾期']))

# Save model
model.save('word2vec_zhwiki_plus_Q.model')
model = Word2Vec.load('word2vec_zhwiki_plus_Q.model')
#print(model.wv.vocab.keys())
