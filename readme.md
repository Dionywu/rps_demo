## 对静态博弈研究的一种思路——基于rps的简单实现
完美信息博弈的对策往往不依赖于特定对手，比如围棋、五子棋、象棋等棋类游戏，信息完全公开，对手的水平有高低之分，己方按照常规思路下的最优策略进行走步就可以达到己方可达到的最优解，alphago也证实了AI在完美信息博弈领域可以战胜人类；而非完美信息博弈(可以简单理解为对局中具有大量的隐藏信息，如德州扑克、斗地主、麻将等)则不同，由于双方观测信息的局限性，很难选择一个最优解。以最简单的非完美信息博弈rps(即石头剪刀布)为例，游戏中并不存在传统意义上稳定最优的策略，或者说最优策略往往要依赖于对手，如当对手永远出石头时，一直出布是最优解(收益恒为正)；而当对手永远出剪刀时，一直出布反而成了最差解(收益恒为负)。
>当然，如果不奢求收益最大化的话，在rps中，纯随机策略是最优解，此时己方收益的期望值为0，尽管不高，但是对手也找不到己方的破绽，这在一定程度上达到了不败之地。——***这里纯随机的策略，指按照(1/3，1/3，1/3)的概率决定出石头、剪刀或布***。

以上一段说明了非完美信息博弈研究的难点，在这类游戏中，想要获得更高的收益是很困难的。

### 这个项目的贡献如下
#### 1.尝试提供一种解决这类问题的通用思想
#### 2. 以简单的游戏rps为例进行演示，验证这种思想是有用的
1. 对手的决策往往与一些“因素”有关，如果能够找到影响对手决策的“因素”(在机器学习领域可以称之为特征提取)，那就可以在一定程度上预知未知的信息，这对帮助己方决策有很积极的影响。
>做出以上假设的前提是对手的决策往往带有主观性: 所有人都知道在石头剪刀布中，选择纯随机是最优的，但几乎没有人真的在石头剪刀布中选择纯随机策略。如有研究者通过分析发现，成年男性更偏好于出石头、而女性则偏好于出剪刀。如果能捕捉到这些特征对玩家做决策的影响，那么在某种程度上就可以做到针对不同对手调整相应策略从而提高自己的收益
2. 对手往往可以按照风格进行分类，也就是说对手的表现并不是孤立的。如果能够找到不同对手之间的相似性，那么就可以将过往针对别的对手的经验运用到与之相似的新对手身上。
>如假定我们现在在玩石头剪刀布，通过过往的经验发现如果一个玩家是男性，20岁出头，来自北方，那么他大概率第一局出石头、第二局出石头、第三局出布；而如果一个玩家是女性，30岁以上，来自南方，那么她大概率第一局出剪刀、第二局出布、第三局出剪刀。现在我们遇到了一个新的对手，发现新对手年纪在28岁，为女玩家，来自南方，那我们就可以根据经验将她分为第二类对手，我们的决策就是(石头、剪刀、石头)。当然，也许我们不能保证三局全胜，但如果能够赢两局的话，那我们的收益也是大于纯随机的，在一定程度上获得了更高的收益

3. 针对某一类对手，我们可以通过虚拟遗憾最小化的算法（即CFR算法）获取相应地针对其的特定策略，这类方法的思想简要说明如下：己方不停地迭代模拟游戏过程，如果在某场对局中己方选择了策略a，获得了正的收益，这说明这种策略是正确的，下次加大选策略a的概率；如果选择了策略b，获得了负收益，则说明策略b是错误的，那么己方会对别的没有选择的策略（如策略a）产生“遗憾感”并累计遗憾值，那么在下一局模拟对弈中，己方会倾向于选择遗憾值高的策略。
>一个显而易见的结论是，在石头剪刀布游戏中，两个玩家互相进行CFR算法的迭代，最后两个人都会选择(1/3,1/3,1/3)的策略，因为迭代次数足够多之后玩家会发现三者的遗憾相当。


### 本项目的代码介绍如下：

1. prepare.py 中包含着前期准备，包括对对手的特征进行提取（选择了最简单的方式）、将对手的特征进行降维操作（本项目中非必要，只是为了方便可视化；但是如果将这种思想迁移到别的较大规模的特征数可能较多的游戏中，降维操作会降低计算量）、将对手进行聚类
2. config_strategy.py 中包含着针对聚类中的每类最具有特征的玩家训练出的策略
3. run_result.py 给定新玩家，算法根据新玩家的特征将其化分为三类中的某一类，同时使用第二步得到的策略进行决策

实验效果表明本方法比纯随机的效果好
