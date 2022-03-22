# 第一步准备
# 将所有石头剪刀布的玩家分为三大类（三大类分别代表其偏好于出石头、偏好于出剪刀或者偏好于出布），进行降维(将)，后聚类

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
def get_strategy(num):   # 随机产生策略 —— 返回一个列表，分别表示出石头，剪刀，布的概率
    opp_li = []
    np.random.seed(42)
    for i in range(num):
        st = np.round(abs(np.random.rand()),2)
        jd = np.round(abs(np.random.uniform(0,1-st)),2)
        bu = np.round((1-st-jd),2)
        opp_li.append([st,jd,bu])
    return opp_li


def plot_pca(li):
    plt.figure()
    plt.scatter(li[:,0:1],li[:,1:2],s=1,c='red')
    plt.pause(0.5)
    plt.close()


def plot_kmeans(li,label):
    plt.figure()
    for i in range(len(li)):
        if(label[i] == 0):
            s1 = plt.scatter(li[i][0],li[i][1],alpha=1,s = 3, c = "red")
        elif(label[i] == 1):
            s2 = plt.scatter(li[i][0],li[i][1],alpha=1,s = 3, c = "green")
        elif(label[i] == 2):
            s3 = plt.scatter(li[i][0],li[i][1],alpha=1,s = 3, c = "blue")   
    plt.legend((s1,s2,s3),("label 0","label 1","label 2"))
    plt.show()

x_train = get_strategy(200)  # 产生n个玩家

# 降维。将对手策略特征由三维（分别表示石头剪刀布的概率）降低到二维，方便观察 
pca = PCA(n_components=2,random_state=42)
x_decom = pca.fit_transform(X = x_train)

# 聚类。将所有对手分为三类。
km = KMeans(n_clusters=3,random_state=42)
km_y = km.fit_predict(x_decom)
kind_3_opp = pca.inverse_transform(km.cluster_centers_)
if __name__ == '__main__': 
    print("随机生成的前十个对手为:")
    print(x_train[:10])
    print(pca.explained_variance_ratio_)
    print("降维后的对手分布点为: ")
    plot_pca(x_decom)
    print("聚类后的三类分布为: ")
    plot_kmeans(x_decom,km_y)
    print("三类的中心点分别为:")
    print(kind_3_opp)
