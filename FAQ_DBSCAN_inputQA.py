import pymongo_custom_module as PymongoCM
import text_preprocess as tp
import pandas as pd

from sklearn.cluster import DBSCAN


Question_list, Q_WS_list, A_WS_list, Category_list, AllField_list = PymongoCM.get_mongodb_row("Library", "FAQ")
FAQ_df = pd.DataFrame({"content_Q": Q_WS_list, "category": Category_list, "content_A": A_WS_list})


FAQ_df["clean_msg_Q"] = FAQ_df.content_Q.apply(tp.text_process)
FAQ_df["clean_msg_A"] = FAQ_df.content_A.apply(tp.text_process)

# Input Data
Q_clean_list = FAQ_df.clean_msg_Q.tolist()
A_clean_list = FAQ_df.clean_msg_A.tolist()
category_list = FAQ_df.category.tolist()


QA_clean_list = []
for i in range(len(Q_clean_list)):
    #print('+', Q_clean_list[i], '=', A_clean_list[i])
    QA_clean_list.append(Q_clean_list[i] + A_clean_list[i])
    
#print(QA_clean_list[:5])
new_df = pd.DataFrame({'Q_clean': QA_clean_list, 'category': category_list})


tfidf_vectorizer, X_tfidf, tfidf_array, tfidf_T_array, terms = tp.tfidf(Q_clean_list, max_df=1.0, min_df=1)



# DBSCAN
dbscan_clf = DBSCAN(eps=0.85, min_samples=2).fit(tfidf_array)

# display by group
frame = pd.DataFrame(tfidf_array)
frame['Cluster'] = dbscan_clf.fit_predict(tfidf_array)
frame['Q'] = Question_list
frame['Q_clean'] = Q_clean_list

print(frame['Cluster'].value_counts())

print("-----------------------------------------------")
print(frame.groupby('Cluster').agg({'Cluster':'count'}))

sectors = frame.groupby('Cluster')
sectors_len = len(sectors)

for ClusterN in range(-1, sectors_len -1, 1):
    print("===== Cluster {} =====".format(ClusterN))
    ClusterN_index = list(sectors.get_group(ClusterN).index)
    print(frame.loc[ClusterN_index].Q)
    
    
print(PymongoCM.get_and_append_mongodb_row("Library", "FAQ", frame))

