#### Class for Dictionary
import pandas as pd
import gensim.corpora as corpora

class Dictionary: # dictionary
    def __init__(self, 
                 ngr, 
                 no_below = 40,
                 no_above = 0.8,
                 keep_n = None,
                 from_pickle = True):
        self.ngr = ngr
        self.from_pickle = from_pickle
        self.id2word = None
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n
        self.name = f"dicfullmc{ngr.ng_min_count}thr{str(ngr.ng_threshold).replace('.', '_')}{ngr.ng_scoring[:3]}nob{self.no_below}noa{str(self.no_above).replace('.', '_')}"
        self.path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/id2word/{self.name}.txt"
        if from_pickle:
            self.load()
        else:
            self.create()
            self.save()
    
    def load(self):
        self.id2word = corpora.Dictionary.load_from_text(self.path)

    def create(self):
        self.id2word = corpora.Dictionary(self.ngr.data_bigrams_trigrams)
        self.id2word.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=self.keep_n)

    def save(self):
        self.id2word.save_as_text(self.path)

        