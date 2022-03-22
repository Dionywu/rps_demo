# 根据1，2步训练出的策略，针对不同的对手选择不同的策略进行对战
from config_strategy import strategy0,strategy1,strategy2,get_score,get_my_action,get_opp_action
from prepare import x_decom,km_y,pca,km,kind_3_opp
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
def get_mystrategy(index):  # 根据不同类对手分别选择第二步中不同的策略
    if(opp_kind_list[index]%3 == 0):
        return strategy0
    if(opp_kind_list[index]%3 == 1):
        return strategy1
    if(opp_kind_list[index]%3 == 2):
        return strategy2

opp = np.array([
    [0.1,0.1,0.8],
    [0.8,0.1,0.1],
    [0.3,0.3,0.4],
    [0.25,0.15,0.6],
    [0.3,0.4,0.3],
    [0.2,0.2,0.6]
])

knclf = KNeighborsClassifier(n_neighbors=5)  # 根据knn算法将对手分为聚的三类中的一类
knclf.fit(x_decom, km_y)
opp_2d = pca.transform(opp.reshape([-1,3]))  # !!!!!!! 这里应该是直接用pca.transform()。我之前用的pca.fit_transform(),他妈的每次都不对，调试了一晚上！！！！
opp_kind_list = knclf.predict(opp_2d)  # 预测opp的分类


li = x_decom
label = km_y
plt.figure()
for i in range(len(li)):
    if(label[i] == 0):
        s1 = plt.scatter(li[i][0],li[i][1],alpha=0.2,s = 2, c = "red")
    elif(label[i] == 1):
        s2 = plt.scatter(li[i][0],li[i][1],alpha=0.2,s = 2, c = "green")
    elif(label[i] == 2):
        s3 = plt.scatter(li[i][0],li[i][1],alpha=0.2,s = 2, c = "blue")
for i in range(len(opp_2d)):
    if(opp_kind_list[i] == 0):
        s4 = plt.scatter(opp_2d[i][0],opp_2d[i][1],alpha=1,s = 4, c = "red")
    elif(opp_kind_list[i] == 1):
        s5 = plt.scatter(opp_2d[i][0],opp_2d[i][1],alpha=1,s = 4, c = "green")
    elif(opp_kind_list[i] == 2):
        s6 = plt.scatter(opp_2d[i][0],opp_2d[i][1],alpha=1,s = 4, c = "blue")
plt.legend((s4,s5,s6),("pred label0","pred label1","pred label2"))
plt.show()


if __name__ == '__main__':
    i = int(input("请输入对手编号:"))
    opp_now = opp[i]
    my_strategy = get_mystrategy(i)
    print("对手的策略为:",opp[i])
    print("对手的分类是:",opp_kind_list[i])
    print("我们的策略为： ",my_strategy)
    print("以下是该算法针对特定对手下1000把的得分：")
    for j in range(10):
        score = 0
        for i in range(1000):
            m_act = get_my_action(my_strategy)
            o_act = get_opp_action(opp_now)
            score += get_score(o_act, m_act)
        print(score)

    print("\n")
    print("以下是随机策略下1000把的得分：")
    for j in range(10):
        score = 0
        for i in range(1000):
            m_act = np.random.choice([0,1,2])
            o_act = get_opp_action(opp_now)
            score += get_score(o_act, m_act)
        print(score)
