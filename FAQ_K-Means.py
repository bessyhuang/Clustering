import pymongo_custom_module as PymongoCM
import text_preprocess as tp
import pandas as pd

from sklearn.cluster import KMeans


Question_list, Q_WS_list, Category_list, AllField_list = PymongoCM.get_mongodb_row("Library", "FAQ")
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

# KMeans (每群內 數量較平均)
kmeans_clf = KMeans(n_clusters=70, init='k-means++', n_init=5, tol=0.1).fit(tfidf_array)
"""
n_init=3  表示進行3次k均值運算後求平均值。一開始迭代，隨機選擇聚類中心點，不同的中心點可能導致不同的收斂效果，故多次運算求平均值的方法可增加穩定性。
tol=0.1   表示中心點移動距離小於0.1時，就認為已經收斂，停止迭代。
verbose=1 表示印出迭代的過程資訊。
"""
print('聚類中心均值向量的總和：', kmeans_clf.inertia_, '\n取得預測結果：', kmeans_clf.labels_)
"""
inertia_ : float，Sum of squared distances of samples to their closest cluster center.
(model_KMeans.labels_ == labels).all() 判斷兩個list是否完全一樣，完全一樣則為True。
labels_ : Labels of each point (表示該文件被分到哪一群)
"""
print(kmeans_clf.labels_.all() == kmeans_clf.predict(tfidf_array).all())
print(kmeans_clf.labels_)
print(kmeans_clf.predict(tfidf_array))

# display by group
frame = pd.DataFrame(tfidf_array)
frame['Cluster'] = kmeans_clf.predict(tfidf_array)
frame['Q'] = Question_list
frame['Q_clean'] = Q_clean_list

print(frame['Cluster'].value_counts())

print("-----------------------------------------------")
print(frame.groupby('Cluster').agg({'Cluster':'count'}))

sectors = frame.groupby('Cluster')
sectors_len = len(sectors)

for ClusterN in range(0, sectors_len, 1):
    print("===== Cluster {} =====".format(ClusterN))
    ClusterN_index = list(sectors.get_group(ClusterN).index)
    print(frame.loc[ClusterN_index].Q)



# 各群的重要單詞 (由大到小排序)
# http://www.cs.uoi.gr/~tsap/teaching/2016-cse012/slides/Intro-to-Clustering.html
order_centroids = kmeans_clf.cluster_centers_.argsort()[:, ::-1]
print(order_centroids, terms[600], terms[489], terms[1062])
"""
cluster_centers_ : array, [n_clusters, n_features]，Coordinates of cluster centers.
上述程式碼的意思：把各別聚類之中心點的不同特徵分量(單字)，由大到小排序，並且把排序後的元素索引儲存到 order_centroids。
order_centroids variable includes all words words' vectors for each cluster and it is ordered
"""

centroids = kmeans_clf.cluster_centers_
print(centroids.argsort().shape)   # 印出 k clusters, all columns (feature_words)
EachCluster_FeatureWords_sorted = centroids.argsort()[:, ::-1]  # 由大到小排序
print(order_centroids.all() == EachCluster_FeatureWords_sorted.all())

EachCluster_FeatureWords_num = 10
for num, centroid in enumerate(EachCluster_FeatureWords_sorted):
    print("--> Cluster %d:" % num, end='')
    for ind in EachCluster_FeatureWords_sorted[num, :EachCluster_FeatureWords_num]:
        print(' %s' % terms[ind], end='')
    print()




"""
# 挑選 KMeans 該分成幾群 (plot the SSE for a range of cluster sizes)
# https://www.kaggle.com/jbencina/clustering-documents-with-tfidf-and-kmeans
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

def find_optimal_clusters(data, max_k):
    SSE = []
    iters = range(2, max_k+1, 10)
    for k in iters:
        km_clf = KMeans(n_clusters=k, init='k-means++', n_init=5, tol=0.1, random_state=1).fit(data)

        # 用來評估集群的個數是否合適，距離越小，代表分群越好。最後選擇【臨界點的集群個數】
        SSE.append(km_clf.inertia_)
        print('Fit {} clusters:'.format(k), km_clf.inertia_)

    # 繪製圖形
    fig, ax = plt.subplots()
    ax.set_xticks(iters)
    ax.set_xticklabels(['集群 ' + str(s) for s in iters])

    frame = pd.DataFrame({'Cluster':iters, 'SSE':SSE})
    ax.plot(frame['Cluster'], frame['SSE'], marker='o')

    ax.set_xlabel('Number of clusters (Cluster Centers)')
    ax.set_ylabel('Inertia (SSE)')
    ax.set_title('SSE (Inertia) by Cluster Center (Number of clusters) Plot', fontsize=12)
    plt.show()

find_optimal_clusters(X_tfidf, 400)


### 各類別的數量統計圖 ###
# FAQ -> 圖書館常見問答集
# Category -> 館務問題類別
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['font.size'] = '15'
fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot(111)

TopN_categories = 10
category_counts = new_df.category.value_counts()[:TopN_categories]
categories = category_counts.index[:TopN_categories]

sns.barplot(x=categories, y=category_counts)
for a, p in enumerate(ax.patches):
    ax.annotate(f'{categories[a]}\n' + format(p.get_height(), '.0f'),
        xy=(p.get_x() + p.get_width() / 2.0, p.get_height()), xytext=(0,-25), size=10,
        color='white' , ha='center', va='center', textcoords='offset points',
        bbox=dict(boxstyle='round', facecolor='none', edgecolor='white', alpha=0.5) )
plt.xlabel('Category')
plt.ylabel('The Number of FAQ')
plt.xticks(size=10) # X軸刻度
plt.title("The number of FAQ by Categories")
plt.show()
"""
