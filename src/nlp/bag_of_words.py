from gensim.models import TfidfModel
from gensim.corpora import MmCorpus
import os
import pathlib
from pathlib import Path
import pickle
import gensim.corpora as corpora

class BagOfWords:
    def __init__(self, ngr = None, dic = None, dicname = None, from_pickle = True):
        self.datafolder = pathlib.Path(__file__).parent.parent.parent / "data/nlp"
        print("Loading bag of words...")
        if not from_pickle:
            self.texts = ngr.data_bigrams_trigrams
            self.id2word = dic.id2word
            self.bow, self.tfidf = self.bow_texts() 
            self.serialize_bow(dic)
            self.dicname = dic.name
            self.yr = ngr.yr_vec
            self.ciks_to_keep = ngr.ciks_to_keep
        else:
            # Assert if dicname is not None:
            
            assert dicname is not None, "dicname must be provided if from_pickle is True"
            self.dicname = dicname
            corpuspath = self.datafolder / f'corpora/{dicname}/corpus_full.mm'
            corpus_info = self.datafolder / f'corpora/{dicname}/corpus_info.pkl'
            self.texts = []
            self.tfidf = []
            with open(corpus_info, "rb") as f:
                _, self.ciks_to_keep, self.yr = pickle.load(f)
            self.bow = MmCorpus(str(corpuspath))
            dicpath = self.datafolder / f'id2word/{dicname}.txt'
            self.id2word = corpora.Dictionary.load_from_text(str(dicpath))

    def serialize_bow(self, dic):
        dicname = dic.name
        dicpath = self.datafolder / f'corpora/{dicname}'
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
        self.path = f"{dicpath}/corpus_full.mm"
        MmCorpus.serialize(str(self.path), self.bow)
        print("Bag of words serialized")

    def bow_texts(self):
        print('Creating corpus and tfidf...')
        corpus = [self.id2word.doc2bow(text) for text in self.texts]
        tfidf = TfidfModel(corpus, id2word = self.id2word)
        print('Corpus and Tfidf created')
        return corpus, tfidf