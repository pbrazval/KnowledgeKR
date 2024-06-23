from sklearn.cluster import KMeans
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import embedding_tools as et
import seaborn as sns
import matplotlib.pyplot as plt

class EmbeddingsHKRModel:
    def __init__(self):
        pass

    def from_clusters(self, embeddings, modelname, nclusters, random_state = 42):
        self.embeddings = embeddings
        self.modelname = modelname
        columns_to_preserve = ['year', 'CIK']
        
        df = self.embeddings.data.copy()

        data_for_clustering = df.drop(columns=columns_to_preserve)

        # Initialize KMeans with 10 clusters
        kmeans = KMeans(n_clusters=nclusters, random_state=random_state)

        # Fit and predict the clusters
        df['cluster'] = kmeans.fit_predict(data_for_clustering)

        df.head()
        df.rename(columns={"cluster": "max_topic"}, inplace=True)
        topic_map = df[['max_topic', 'year', 'CIK']]
        # Save topic_map as a CSV to /Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/embeddings_km10
        topic_map.to_csv(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}/topic_map_2006_2022.csv", index=False)
        self.topic_map = topic_map
        return self
    
    def from_topic_similarity(self, embeddings, modelname, term, to_csv = True):
        self.embeddings = embeddings
        self.modelname = modelname
        term_embedding = np.array(et.get_text_embeddings(term)).reshape(1, -1)
        
        # Function to calculate cosine similarity between row vector A and term_embedding B
        def calculate_cosine_similarity(row):
            vector_a = row.values.reshape(1, -1)
            return cosine_similarity(vector_a, term_embedding)[0, 0]
        # Create a dataframe df as a copy of the embeddings data:
        df = self.embeddings.data.copy()

        # Apply the function to each row in the DataFrame
        df['term_cs'] = df.iloc[:, 0:1536].apply(calculate_cosine_similarity, axis=1)

        # Plot a histogram of the cosine similarity values
        # Rename term_cs to topic_kk:
        df.rename(columns={"term_cs": "topic_kk"}, inplace=True)
        # Save the updated DataFrame to a CSV file
        #df.rename(columns={"cluster": "max_topic"}, inplace=True)
        topic_map = df[['topic_kk', 'year', 'CIK']]
        # Save topic_map as a CSV to /Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/embeddings_km10
        if to_csv:
            output_dir = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}"
            os.makedirs(output_dir, exist_ok=True)
            topic_map.to_csv(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}/topic_map_2006_2022.csv", index=False)
        self.modelname = modelname
        self.topic_map = topic_map
        return self



    @staticmethod
    def plot_cosine_similarity(words):
        # E.g. words = ["innovation", "intellectual property", "patent", "knowledge capital", "clinical trial", "software"]
        # Get embeddings for the provided words
        embeddings = [et.get_text_embeddings(word) for word in words]

        # Stack embeddings into a matrix
        embeddings_matrix = np.vstack(embeddings)

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(embeddings_matrix)

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(cosine_sim, dtype=bool))

        # Create a heatmap with the cosine similarity, masking the upper triangle
        sns.heatmap(cosine_sim, annot=True, xticklabels=words, yticklabels=words, cmap='viridis', mask=mask)
        plt.title('Cosine Similarity between Word Embeddings')
        plt.show()

        # Get embeddings for the words "innovation", "intellectual property", "patent", "knowledge capital", "clinical trial", and "software"
        # inno_et = et.get_text_embeddings("innovation")
        # term_et = et.get_text_embeddings("intellectual property")
        # patent_et = et.get_text_embeddings("patent")
        # knowledge_capital_et = et.get_text_embeddings("knowledge capital")
        # clinical_trial_et = et.get_text_embeddings("clinical trial")
        # software_et = et.get_text_embeddings("software")

        # # Stack embeddings into a matrix
        # embeddings_matrix = np.vstack([inno_et, term_et, patent_et, knowledge_capital_et, clinical_trial_et, software_et])

        # # Calculate cosine similarity
        # cosine_sim = cosine_similarity(embeddings_matrix)

        # # Create a mask for the upper triangle
        # mask = np.triu(np.ones_like(cosine_sim, dtype=bool))

        # # Create a heatmap with the cosine similarity, masking the upper triangle
        # words = ["innovation", "intellectual property", "patent", "knowledge capital", "clinical trial", "software"]
        # sns.heatmap(cosine_sim, annot=True, xticklabels=words, yticklabels=words, cmap='viridis', mask=mask)
        # plt.title('Cosine Similarity between Word Embeddings')
        # plt.show()