# nlp-learning-notebook

## 理解word2vec

参考：
https://medium.com/explore-artificial-intelligence/word2vec-a-baby-step-in-deep-learning-but-a-giant-leap-towards-natural-language-processing-40fe4e8602ba
https://zhuanlan.zhihu.com/p/53194407

word2vec模型又叫“词嵌入”，是词的向量表征。在自然语言预处理阶段，需要找到一种恰当的编码方式来代表某个词，相较于其他编码方式，比如one-hot，word2vec能够得到低维且稠密的向量，这非常有利于后续深度学习，抓取语义。
词向量模型在nlp发展中有着久远的历史，其思想是将词嵌入到连续的向量空间中使得语义相似的词在空间里被映射为空间当中距离接近的点。它们都基于分布假设，即出现在相同文本里的词具有相同的语义。实现这个原理的路径有两条：基于频数和基于预测。word2vec 是一种计算非常高效的基于预测的模型，能够从raw text里学习获得词向量。
word2vec训练获得词向量的方式有两种，一种是continous bag-of-words model(CBOW)，另一种叫skip-gram model. 前者是通过给定相邻的词预测目标词，后者是通过目标词预测与之相邻的词。通常采取后面一种方式。以下主要介绍skip-gram模型。