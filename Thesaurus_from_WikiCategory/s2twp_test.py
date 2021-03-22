from opencc import OpenCC
import os

path_input = '/home/bessyhuang/圖片/Thesaurus_for_FAQws/'
path_output = '/home/bessyhuang/圖片/Thesaurus_for_FAQws_tw/'
files = os.listdir(path_input)
#print(files)

cc = OpenCC('s2twp')
#text = '文渊阁'
#print(cc.convert(text))


for doc in files:
    print(doc)
    with open(path_input + doc, 'r', encoding="utf-8") as f_in:
        text_list = f_in.read().splitlines()
    #print('===', text_list)
    
    new_text_list = []
    for w in text_list:
        new_text_list.append(cc.convert(w))
    #print(new_text_list)
    
    f_out = open(path_output + doc, 'w', encoding="utf-8")
    for w in new_text_list:
        f_out.write(w + '\n')
    f_out.close()
    
