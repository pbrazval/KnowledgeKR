import gensim
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.models import HdpModel
from gensim.models import LdaMulticore
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
from .nlp import *


class NLPRiskMeasure:
    def __init__(self, redo = False):
        self.redo = redo

class TopicModel(NLPRiskMeasure):
    def __init__(self, bow, redo = False):
        self.corpus = bow.bow
        self.id2word = bow.id2word
        self.dicname = bow.dic.name
        self.yr = bow.ngr.yr
        self.ciks_to_keep = bow.ngr.ciks_to_keep
        self.redo = redo


class LDA(TopicModel):
    def __init__(self, bow, num_topics, modelname = None, redo = False):
        super().__init__(self, bow, redo)
        self.num_topics = num_topics

        if modelname is not None:
            self.modelname = f"{bow.dic.dicname}_{num_topics}_t"
        else:
            self.modelname = modelname
        self.modelpath = datapath(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/code/step_by_step/{self.modelname}")
        if self.redo:
            self.model, self.topics_per_doc = self.create_model()
        else:
            self.model, self.topics_per_doc = self.retrieve_model()
        #self.print_coherence()
        #self.visualize_topics()
        #self.create_topic_map()

    def visualize_topics(self):
        model = self.model
        corpus = self.corpus
        id2word = self.id2word
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim_models.prepare(model, corpus, id2word, mds = "mmds", R = 50)
        vis

    def retrieve_model(self):
        print(f"Retrieving model {self.modelname}...")
        model = LdaModel.load(self.modelpath)
        topics_per_doc = [model[unseen_doc] for unseen_doc in self.corpus]
        return model,topics_per_doc

    def create_model(self):
        print(f"Making model {self.modelname}...")
        model = gensim.models.LdaMulticore(corpus=self.corpus,
                                               id2word=self.id2word,
                                               num_topics=self.num_topics,
                                               random_state=100, passes = 10) #update_every=1,chunksize=100,passes=10,#alpha="auto"
        model.save(self.modelpath)
        topics_per_doc = [model.get_document_topics(doc) for doc in self.corpus]
        print(f"Model {self.modelname} created.")
        return model, topics_per_doc    
    
    def print_coherence(self):
        model = self.model
        corpus = self.corpus
        num_topics = self.num_topics
        cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print(f"Number of topics: {num_topics}. Coherence: {coherence}")
        return None

    def create_topic_map(model, topics_per_doc, yr, ciks_to_keep, modelname):
        try:
            k = model.num_topics
        except:
            k = model.m_T
        
        cik_list = ciks_to_keep.values
        topic_probs = {f"topic_{i}": [] for i in range(k)}
        max_topic = [max(doc, key = lambda x:x[1])[0] for doc in topics_per_doc]
        # Iterate over each list of tuples in the topic list
        for doc in topics_per_doc:
            # Initialize a dictionary to hold the probabilities for this topic
            topic_dict = {f"topic_{i}": 0.0 for i in range(k)}
            for tup in doc:
                topic_dict[f"topic_{tup[0]}"] = tup[1]
            # Append the topic probabilities to the overall topic_probs dictionary
            for i in range(k):
                topic_probs[f"topic_{i}"].append(topic_dict[f"topic_{i}"])

        # Create a Pandas DataFrame using the topic_probs dictionary and the cik_list
        df = pd.DataFrame.from_dict(topic_probs)
        df['max_topic'] = max_topic
        df['CIK'] = cik_list
        df['year'] = yr
        
        if not os.path.exists(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}"):
            os.makedirs(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}")
        if isinstance(yr, list):
            df.to_csv(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}/topic_map_{min(yr)}_{max(yr)}.csv", index=False)
        else:
            df.to_csv(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}/topic_map_{yr}.csv", index=False)
        return df
    
class HDP(TopicModel):
    def __init__(self, bow, cutoff = 6, modelname = None, redo = False):
        super().__init__(self, bow, redo)
        self.modelname = f"{bow.dic.dicname}_hdp"
        self.modelpath = datapath(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/code/step_by_step/{self.modelname}")
        self.cutoff = cutoff
        if self.redo:
            self.model, self.topics_per_doc = self.create_model()
        else:
            self.model, self.topics_per_doc = self.retrieve_model()
    
    def create_model(self):

        print(f"Making HDP model {self.modelname}...")
        hdp_model = gensim.models.HdpModel(corpus=self.corpus,
                                id2word=self.id2word,
                                random_state=100) #update_every=1,chunksize=100,passes=10,#alpha="auto"
        hdp_model.save(f"/Users/pedrovallocci/Documents/PhD (local)/Research/Github/KnowledgeKRisk_10Ks/text/{modelname}_hdp/hdp_model")
        df = self.create_topic_map(self)
        topics_per_doc = [hdp_model.get_document_topics(doc) for doc in self.corpus]
        print(f"Model {self.modelname} created.")
        return hdp_model, topics_per_doc  

    def retrieve_model(self):
        print(f"Retrieving model {self.modelname}...")
        model = HdpModel.load(self.modelpath)
        topics_per_doc = [model[unseen_doc] for unseen_doc in self.corpus]
        return model, topics_per_doc  
    
    def create_topic_map(self):
        ciks_to_keep = self.bow.ngr.ciks_to_keep
        yr = self.bow.ngr.yr
        hdp_model = self.model
        dicname = self.dicname
        corpus = self.corpus
        cutoff = self.cutoff
        modelname = f"{dicname}_hdp"
        topics_per_doc = [hdp_model[unseen_doc] for unseen_doc in corpus]
        try:
            k = hdp_model.num_topics
        except:
            k = hdp_model.m_T

        cik_list = ciks_to_keep.values
        topic_probs = {f"topic_{i}": [] for i in range(k)}
        max_topic = [max(doc, key = lambda x:x[1])[0] if len(doc) > 0 else None for doc in topics_per_doc]
        max_topic = [x if x <= cutoff else 999 for x in max_topic]

        # Iterate over each list of tuples in the topic list
        for doc in topics_per_doc:
        # Initialize a dictionary to hold the probabilities for this topic
            topic_dict = {f"topic_{i}": 0.0 for i in range(k)}
            for tup in doc:
                topic_dict[f"topic_{tup[0]}"] = tup[1]
        # Append the topic probabilities to the overall topic_probs dictionary
            for i in range(k):
                topic_probs[f"topic_{i}"].append(topic_dict[f"topic_{i}"])

        # Create a Pandas DataFrame using the topic_probs dictionary and the cik_list
        df = pd.DataFrame.from_dict(topic_probs)
        df['max_topic'] = max_topic
        df['CIK'] = cik_list
        df['year'] = yr

        if not os.path.exists(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}"):
            os.makedirs(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}")
        if isinstance(yr, list):
            df.to_csv(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}/topic_map_{min(yr)}_{max(yr)}.csv", index=False)
        else:
            df.to_csv(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}/topic_map_{yr}.csv", index=False)
        return df
        