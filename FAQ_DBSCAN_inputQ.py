import pymongo_custom_module as PymongoCM
import text_preprocess as tp
import pandas as pd

from sklearn.cluster import DBSCAN


Question_list, Q_WS_list, A_WS_list, Category_list, AllField_list = PymongoCM.get_mongodb_row("Library", "FAQ")
FAQ_df = pd.DataFrame({"content": Q_WS_list, "category": Category_list})

print(FAQ_df)

FAQ_df["clean_msg"] = FAQ_df.content.apply(tp.text_process)

# Input Data
Q_clean_list = FAQ_df.clean_msg.tolist()
category_list = FAQ_df.category.tolist()

new_df = pd.DataFrame({'Q_clean': Q_clean_list, 'category': category_list})
print(new_df)

tfidf_vectorizer, X_tfidf, tfidf_array, tfidf_T_array, terms = tp.tfidf(Q_clean_list, max_df=0.8, min_df=2)
print()

# DBSCAN
dbscan_clf = DBSCAN(eps=0.95, min_samples=5).fit(tfidf_array)
print(all(dbscan_clf.labels_ == dbscan_clf.fit_predict(tfidf_array)))
print(dbscan_clf.labels_)
print(dbscan_clf.fit_predict(tfidf_array))

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

