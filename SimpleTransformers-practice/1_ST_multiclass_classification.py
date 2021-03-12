### Colab GPU Testing & Setting & Installing ###
"""
# Returns the name of a GPU device if available or the empty string.

import tensorflow as tf
print(tf.test.gpu_device_name())


# Current available GPUs in tensorflow

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Mounted at Google Drive

from google.colab import drive
drive.mount('/content/drive/')

import os
os.chdir("/content/drive/My Drive/Colab Notebooks")
!ls


# Install SimpleTransformers

!pip3 install simpletransformers
"""


### Encode label (Category) ###
"""
from collections import defaultdict
import csv


columns = defaultdict(list)
with open('4975_LibraryFAQ_raw.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print('-- row --', row)
        for (k,v) in row.items():
            #print(k, v)
            columns[k].append(v)

#print(columns['Category'], columns['LibraryName'])

index = 0
Category_dict = defaultdict(dict)
for row in columns['Category']:
    if row in Category_dict:
        pass
    else:
        Category_dict[row] = index
        index += 1
#print(Category_dict)

with open('Encode_4975_LibraryFAQ.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['_id', 'Question', 'Category'])
    for i in range(len(columns['_id'])):
        cat_index = Category_dict[columns['Category'][i]]
        writer.writerow([columns['_id'][i], columns['Question'][i], cat_index])

with open('Encode_4975_LibraryFAQ.txt', 'a', encoding='utf-8')  as f:
    for key in Category_dict:
        f.write(str(Category_dict[key]) + '\t' + key + '\n')
"""


### Start! ###

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# Preparing raw data
data = pd.read_csv("Encode_4975_LibraryFAQ.csv", usecols=[1,2], header=0)
data.columns = ['text', 'labels']

print(data.iloc[[100]])

#data['labels'] = pd.to_numeric(data.labels, errors="raise")
#print(data['labels'].dtype)

data = data[:].values
print(data)


# Train data
train_data = data

train_df = pd.DataFrame(train_data)
train_df.columns = ['text', 'labels']

# Eval data
eval_data = data

eval_df = pd.DataFrame(eval_data)
eval_df.columns = ['text', 'labels']

# Optional model configuration
model_args = ClassificationArgs(
    num_train_epochs=15,
    reprocess_input_data=True,
    overwrite_output_dir=True
)   # , max_seq_length=512

# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-chinese', num_labels=490, args=model_args)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["何時開館？"])
print(predictions, raw_outputs)

# Save the trained model: https://www.philschmid.de/bert-text-classification-in-a-different-language
import os
import tarfile

def pack_model(model_path='',file_name=''):
    files = [files for root, dirs, files in os.walk(model_path)][0]
    with tarfile.open(file_name + '.tar.gz', 'w:gz') as f:
        for file in files:
            f.add(f'{model_path}/{file}')

pack_model('./','ST_ClassificationModel_20210312')


### Load the model and predict a real example ###
from simpletransformers.classification import ClassificationModel

model = ClassificationModel(model_type='bert', model_name='./outputs/', num_labels=490)

# Category Dict
from collections import defaultdict
import csv


DICT_key_INDEX = defaultdict(dict)

with open('./Encode_4975_LibraryFAQ.txt', 'r', encoding='utf-8') as f:
    content = f.readlines()
    for row in content:
        index, category = row.split('\t')
        DICT_key_INDEX[eval(index)] = category.replace('\n', '')
print(DICT_key_INDEX)



"""
### Predict ###
predictions, raw_outputs = model.predict(["何時開館？"])
print("==>", "何時開館？", predictions, DICT_key_INDEX[predictions[0]])


### Practice ###
Sample_DICT_key_category = {
    '借閱規則及服務': 0, '館藏資料位置': 1,
    '如何查詢學位論文': 2, '圖書資料借閱與歸還': 3
}
Sample_DICT_key_index = dict((v,k) for k,v in Sample_DICT_key_category.items())
#print('{ key: index, value: category } -->', Sample_DICT_key_index)
"""

# ========================================
### 重新預測原本的 Question 該分到哪一類 ###
import pandas as pd

# Preparing raw data
data = pd.read_csv("Encode_4975_LibraryFAQ.csv", usecols=[1,2], header=0)
data.columns = ['text', 'labels']


#data['labels'] = pd.to_numeric(data.labels, errors="raise")
#print(data['labels'].dtype)

data = data[:].values
#print(data)


# Test data
test_data = data

test_df = pd.DataFrame(test_data)
test_df.columns = ['text', 'labels']

# Compare RAW & NEW label
raw_predict_same = []
raw_predict_diff = []
raw_predict_all = []

for i in range(len(test_data)):
    print('###', test_data[i])
    predictions, raw_outputs = model.predict([test_data[i][0]])

    raw_predict_all.append([test_data[i][0], predictions[0], DICT_key_INDEX[predictions[0]]])
    
    if test_data[i][1] == predictions[0]:
        raw_predict_same.append([test_data[i][0], test_data[i][1], DICT_key_INDEX[predictions[0]]])
    else:
        raw_predict_diff.append([test_data[i][0], predictions[0], DICT_key_INDEX[predictions[0]]])
        print("==>", test_data[i][0], predictions, DICT_key_INDEX[predictions[0]])


name_attribute = ['Question', 'Predict_CategoryIndex', 'Predict_Category']

writerCSV_all = pd.DataFrame(columns=name_attribute, data=raw_predict_all)
writerCSV_all.to_csv('./Predict_all.csv', encoding='utf-8')

writerCSV1 = pd.DataFrame(columns=name_attribute, data=raw_predict_diff)
writerCSV1.to_csv('./Predict_diff.csv', encoding='utf-8')

writerCSV2 = pd.DataFrame(columns=name_attribute, data=raw_predict_same)
writerCSV2.to_csv('./Predict_same.csv', encoding='utf-8')

