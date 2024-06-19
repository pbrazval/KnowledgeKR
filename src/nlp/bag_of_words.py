from gensim.models import TfidfModel
from gensim.corpora import MmCorpus
import os

class BagOfWords:
    def __init__(self, ngr, dic):
        self.ngr = ngr
        self.dic = dic
        self.texts = ngr.data_bigrams_trigrams
        self.id2word = dic.id2word
        self.bow, self.tfidf = self.bow_texts() 
        self.serialize_bow(dic)

    def serialize_bow(self, dic):
        dicname = dic.name
        dicpath = f'/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/corpora/{dicname}'
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
        self.path = f"{dicpath}/corpus_full.mm"
        MmCorpus.serialize(self.path, self.bow)
        print("Bag of words serialized")

    def bow_texts(self):
        print('Creating corpus and tfidf...')
        corpus = [self.id2word.doc2bow(text) for text in self.texts]
        tfidf = TfidfModel(corpus, id2word = self.id2word)
        print('Corpus and Tfidf created')
        return corpus, tfidf