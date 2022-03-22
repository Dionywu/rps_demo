# 第二步准备：
# 在进行实际对战之前进行训练，分别对上一步的三类玩家中的典型玩家进行分别训练，产生三种策略

import numpy as np
from prepare import pca,km,kind_3_opp,x_decom,km_y
sjb = [0,1,2]  # 石头,剪刀,布

def get_opp_action(opp_strategy_prob_list):   # 根据对手的策略概率获得对手的下一步可能的出法
    return np.random.choice(sjb,p = opp_strategy_prob_list)
def get_my_action(my_strategy_prob_list):  # 根据我的策略概率列表得到我的下一步出法
    return np.random.choice(sjb, p = my_strategy_prob_list)
def get_score(opp_action,my_action):  # 两边同时做动作，获得分数
    if((opp_action == 0 and my_action == 1) or (opp_action == 1 and my_action == 2) or (opp_action == 2 and my_action == 0)):
        return -1
    elif((opp_action == 0 and my_action == 2) or (opp_action == 1 and my_action == 0) or (opp_action == 2 and my_action == 1)):
        return 1
    else:
        return 0
def get_strategy_by_regretlist(the_opp):  # 获得对手的策略，通过cfr算法获得自己应对的策略
    regretlist = np.array([1,1,1])
    # the_opp = opp[0]
    for i in range(10000):
        o_act = get_opp_action(the_opp)
        m_act = get_my_action(regretlist/regretlist.sum())
        if(get_score(o_act, m_act)<0):
            if(m_act == 1):
                regretlist[2] += 1
            elif(m_act == 2):
                regretlist[0] += 1
            elif(m_act == 0):
                regretlist[1] += 1
    my_strategy = regretlist/regretlist.sum()
    return my_strategy


# 针对第一步准备中的三类的核心点分别训练出策略
strategy0 = get_strategy_by_regretlist(kind_3_opp[0])  
strategy1 = get_strategy_by_regretlist(kind_3_opp[1])
strategy2 = get_strategy_by_regretlist(kind_3_opp[2])
if __name__ == '__main__':
    print("通过第一步我们得到的三类对手的典型策略分别为:")
    print(kind_3_opp)
    print("针对以上三种对手，我们的策略分别为:")
    print(strategy0,"\n",strategy1,"\n",strategy2)




