## 评价语言模型
* perplexity
  
困惑度的定义：
$$PP(W) = P(w_1,w_2,...w_N)^{-\frac{1}{N}}$$
基本思想：句子出现的概率越大，困惑度越小，语言模型越好。

* BLEU 机器翻译的评估指标

[BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf)

BLEU, or the Bilingual Evaluation Understudy, is a score for comparing a candidate translation of text to one or more reference translations.

