from gensim import corpora, models
import math
import pandas as pd
from database import Mysql
import nltk
import jieba
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models, similarities
import numpy as np
import jieba
import jieba.posseg as jp
import re
from gensim import corpora, models
import matplotlib.pyplot as plt
# 停止词
stopwords = nltk.corpus.stopwords.words(r'stopword.txt')
def length_bigger_than_10(text):
    if len(text)>=5:
        return True
    else:
        return False

def clean_text(text):


    wordlist = jieba.lcut(text)
    #去除停用词和长度小于2的词语
    wordlist = [w for w in wordlist if w not in stopwords and len(w)>2]
    #将中文数据组织成类似西方于洋那样，词语之间以空格间隔
    document =  " ".join(wordlist)
    return document

def get_train_data():
    mysql = Mysql(user="policy", password="policyAdmin", db="dbpolicy", host="121.36.33.190")
    result1 = mysql.select("medical_policy", "*", "main_text IS NOT NULL limit 400000")
    train = []
    for policy in result1:
        text = policy['main_text'].replace("[^\u4e00-\u9fa5]", "")
        if length_bigger_than_10(text):
            corpus = clean_text(text)
            train.append([word for word in corpus.split(' ')])
    return train


def ldamodel(num_topics):
    train = get_train_data()
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in
              train]  # corpus里面的存储格式（0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)
    corpora.MmCorpus.serialize('corpus.mm', corpus)
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, random_state=1,
                          num_topics=num_topics)  # random_state 等价于随机种子的random.seed()，使每次产生的主题一致

    topic_list = lda.print_topics(num_topics, 10)
    print("主题的单词分布为：\n")
    for topic in topic_list:
        print(topic)
    return lda, dictionary

def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    print('the info of this ldamodel: \n')
    print('num of topics: %s' % num_topics)
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = []
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = []
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0  # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0
        for word_id, num in dict(doc).items():
            prob_word = 0.0
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic * prob_topic_word
            prob_doc += math.log(prob_word)  # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum / testset_word_num)  # perplexity = exp(-sum(p(d)/sum(Nd))
    print("模型困惑度的值为 : %s" % prep)
    return prep


def graph_draw(topic, perplexity):  # 做主题数与困惑度的折线图
    x = topic
    y = perplexity
    plt.plot(x, y, color="red", linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")
    plt.show()


if __name__ == '__main__':
    a = range(1, 20, 1)  # 主题个数
    p = []
    for num_topics in a:
        lda, dictionary = ldamodel(num_topics)
        corpus = corpora.MmCorpus('corpus.mm')
        testset = []
        for c in range(int(corpus.num_docs / 100)):  # 如何抽取训练集
            testset.append(corpus[c * 100])
        prep = perplexity(lda, testset, dictionary, len(dictionary.keys()), num_topics)
        p.append(prep)

    graph_draw(a, p)
