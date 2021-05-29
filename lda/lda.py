import pandas as pd
from database import Mysql
import nltk
from gensim import corpora, models, similarities
import numpy as np
import jieba
import math
import matplotlib.pyplot as plt

import pyLDAvis.sklearn
import pyLDAvis.gensim_models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
from string import digits


NUM = 100000
K_CLUSTER = 100
FILENAME = 'lda100.html'

# 加载停止词
stopwords = nltk.corpus.stopwords.words(r'stopword.txt')
# 后续需要添加自定义停用词

# 去除文本长度小于10的文本
def length_bigger_than_10(text):
    if len(text)>=10:
        return True
    else:
        return False


# 文本清洗
def clean_text(text):
    # 需要自定义在医疗政策领域的词典
    wordlist = jieba.lcut(text)
    # 去除停用词和长度小于2的词语
    wordlist = [w for w in wordlist if w not in stopwords and len(w)>2]
    # 将中文数据组织成类似西方于洋那样，词语之间以空格间隔
    document =  " ".join(wordlist)
    return document


# 连接数据库
def get_data():
    mysql = Mysql(user="policy", password="policyAdmin", db="dbpolicy", host="121.36.33.190")
    return mysql.select("medical_policy", "*", f"main_text IS NOT NULL limit {NUM}")


# 打印结果
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# 计算困惑值
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
    plt.savefig('250-500250-500.png')  # 文件名需要改
    plt.show()


# 调用sklearn的LDA包
def lda_sklearn():
    corpus_list = []
    train = []
    i = 0
    # 得到语料库和训练集
    for policy in get_data():
        print(i)
        i+=1
        text = policy['main_text'].replace("[^\u4e00-\u9fa5]", "")  # 去除非汉字
        remove_digits = str.maketrans('', '', digits)  # 去除数字
        text = text.translate(remove_digits)
        if length_bigger_than_10(text):  # 文本长度需要大于10
            corpus = clean_text(text)  # 分词，去除停用词
            corpus_list.append(corpus)
            train.append([word for word in corpus.split(' ')])
    # 将得到的预料库转换成词频矩阵
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(corpus_list)
    # 训练LDA模型
    lda_model = LatentDirichletAllocation(n_components=K_CLUSTER, random_state=888)
    lda_model.fit(doc_term_matrix)

    # 打印结果
    n_top_words = 12
    tf_feature_names = vectorizer.get_feature_names()
    print_top_words(lda_model, tf_feature_names, n_top_words)

    # 可视化展示
    data = pyLDAvis.sklearn.prepare(lda_model, doc_term_matrix, vectorizer)
    pyLDAvis.save_html(data, FILENAME)


# 调用gensim的LDA包
def lda_gensim():
    corpus_list = []
    train = []
    i = 0
    # 得到语料库和训练集
    for policy in get_data():
        print(i)
        i += 1
        text = policy['main_text'].replace("[^\u4e00-\u9fa5]", "")
        remove_digits = str.maketrans('', '', digits)  # 去除数字
        text = text.translate(remove_digits)
        if length_bigger_than_10(text):
            corpus = clean_text(text)
            corpus_list.append(corpus)
            train.append([word for word in corpus.split(' ')])
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    # corpus_tfidf = models.TfidfModel(corpus)[corpus]
    lda = models.LdaModel(corpus, num_topics=K_CLUSTER, id2word=dictionary,random_state=200)

    # 打印结果
    for i in range(K_CLUSTER):
        print(lda.print_topic(i))

    # 可视化展示
    data = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(data, FILENAME)

# LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
#              evaluate_every=-1, learning_decay=0.7,
#              learning_method='batch', learning_offset=10.0,
#              max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
#              n_components=5, n_jobs=None, perp_tol=0.1,
#              random_state=888, topic_word_prior=None,
#              total_samples=1000000.0, verbose=0)


#
def cal_prep(start=250, end=500, sep=1):
    '''
    :param start: 表示主题数开始值
    :param end: 表示主题数结束值
    :param sep: 表示循环间隔，默认为1
    :return:
    '''
    corpus_list = []
    train = []
    i = 0
    # 得到语料库和训练集
    for policy in get_data():
        print(i)
        i += 1
        text = policy['main_text'].replace("[^\u4e00-\u9fa5]", "")
        remove_digits = str.maketrans('', '', digits)  # 去除数字
        text = text.translate(remove_digits)
        if length_bigger_than_10(text):
            corpus = clean_text(text)
            corpus_list.append(corpus)
            train.append([word for word in corpus.split(' ')])
    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    corpora.MmCorpus.serialize('corpus.mm', corpus)
    corpus = corpora.MmCorpus('corpus.mm')
    a = range(start, end, sep)  # 遍历主题个数
    p = []
    for num_topics in a:
        lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=200)
        testset = []
        for c in range(int(corpus.num_docs / 100)):  # 如何抽取训练集
            testset.append(corpus[c * 100])
        prep = perplexity(lda, testset, dictionary, len(dictionary.keys()), num_topics)
        p.append(prep)
    graph_draw(a, p)


if __name__ == '__main__':
    # lda_sklearn()
    lda_gensim()
    # cal_prep() #
