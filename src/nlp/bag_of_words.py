from gensim.models import TfidfModel

class BagOfWords:
    def __init__(self, ngr, dic):
        self.ngr = ngr
        self.dic = dic
        self.texts = ngr.data_bigrams_trigrams
        self.id2word = dic.id2word
        self.bow, self.tfidf = self.bow_texts() 

    def bow_texts(self):
        print('Creating corpus and tfidf...')
        corpus = [self.id2word.doc2bow(text) for text in self.texts]
        tfidf = TfidfModel(corpus, id2word = self.id2word)
        print('Corpus and Tfidf created')
        return corpus, tfidf