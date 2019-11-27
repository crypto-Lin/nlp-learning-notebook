# nlp-learning-notebook

## 理解word2vec

参考：

[https://medium.com/explore-artificial-intelligence/word2vec-a-baby-step-in-deep-learning-but-a-giant-leap-towards-natural-language-processing-40fe4e8602ba](https://medium.com/explore-artificial-intelligence/word2vec-a-baby-step-in-deep-learning-but-a-giant-leap-towards-natural-language-processing-40fe4e8602ba)

[https://zhuanlan.zhihu.com/p/53194407](https://zhuanlan.zhihu.com/p/53194407)

word2vec模型又叫“词嵌入”，是词的向量表征。在自然语言预处理阶段，需要找到一种恰当的编码方式来代表某个词，相较于其他编码方式，比如one-hot，word2vec能够得到低维且稠密的向量，这非常有利于后续深度学习，抓取语义。

词向量模型在nlp发展中有着久远的历史，其思想是将词嵌入到连续的向量空间中使得语义相似的词在空间里被映射为空间当中距离接近的点。它们都基于分布假设，即出现在相同文本里的词具有相同的语义。实现这个原理的路径有两条：基于频数和基于预测。word2vec 是一种计算非常高效的基于预测的模型，能够从raw text里学习获得词向量。

word2vec训练获得词向量的方式有两种，一种是continous bag-of-words model(CBOW)，另一种叫skip-gram model. 前者是通过给定相邻的词预测目标词，后者是通过目标词预测与之相邻的词。通常采取后面一种方式。以下主要介绍skip-gram模型。

我们的目标是希望求得每个词的低维向量表示，记为$\theta$，损失函数的设计是使得给定的目标词$w_t$预测生成的相邻的$c$个词的概率在全局预料上取得极大，通过极大似然估计求得目标参数$\theta$:

$$argmax_{\theta}\frac{1}{T}\sum_{t=1}^{T}\sum_{j!=0,j\in c}logP(w_{t+j}|w_t;\theta)$$
$c$代表了一个滑动窗口，设定了模型的预测范围，取$log$之前，似然函数写成：
$$argmax_{\theta}\prod_{t=1}^{T}\prod_{j!=0,j\in c}p(w_{t+j}|w_t;\theta)$$
即该似然函数的估计有一个前提，就是不考虑窗口内的词序列问题。现在要思考的问题是，如何定义$p(w_{t+j}|w_t;\theta)$，即概率表达式要以$\theta$为参数，以便于之后极大化求取，可以定义：
$$p(w_i|w_t;\theta) = \frac{exp(\theta w_i)}{\sum_texp(\theta w_t)}$$
此处$\theta$也被称为embedding lookup matrix，如果$w_i$是$n$维基于one-hot的编码，$\theta$便是$n*k$维matrix，可以将原始编码重新映射为长度为$k$目标编码。从结构上看来，这就是一个简单的3层神经网络：

1. 输入一个单词，训练网络获得它的邻近单词
2. 隐藏层的输出就是输入单词的低维嵌入表示

问题：能根据理解手写一个word2vec的词预训练器吗？




