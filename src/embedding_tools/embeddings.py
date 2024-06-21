import pandas as pd
import numpy as np
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import embedding_tools as et
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

class Embeddings:
    def __init__(self, from_pickle = True):
        
        self.from_pickle = from_pickle
        if not from_pickle:
            self.data = Embeddings.rebuild()
        else:
            self.data = Embeddings.from_pickle()
        # Preserve the columns 'year' and 'CIK'
        self.calculate_clusters(self)

    def calculate_topic_clusters(self, nclusters, modelname, random_state = 42):
        columns_to_preserve = ['year', 'CIK']
        data_for_clustering = self.data.drop(columns=columns_to_preserve)

        # Initialize KMeans with 10 clusters
        kmeans = KMeans(n_clusters=nclusters, random_state=random_state)

        # Fit and predict the clusters
        self.data['cluster'] = kmeans.fit_predict(data_for_clustering)

        self.data.head()
        self.data.rename(columns={"cluster": "max_topic"}, inplace=True)
        topic_map = self.data[['max_topic', 'year', 'CIK']]
        # Save topic_map as a CSV to /Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/embeddings_km10
        topic_map.to_csv(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}/topic_map_2006_2022.csv", index=False)

    def calculate_topic_similarity(self, term, modelname):
        ip_embedding = np.array(et.get_text_embeddings(term)).reshape(1, -1)

        # Function to calculate cosine similarity between row vector A and ip_embedding B
        def calculate_cosine_similarity(row):
            vector_a = row.values.reshape(1, -1)
            return cosine_similarity(vector_a, ip_embedding)[0, 0]

        # Apply the function to each row in the DataFrame
        self.data['ip_cs'] = self.data.iloc[:, 0:1536].apply(calculate_cosine_similarity, axis=1)

        # Display the updated DataFrame
        print(self.data.head())

        # Plot a histogram of the cosine similarity values
        # Rename ip_cs to topic_kk:
        self.data.rename(columns={"ip_cs": "topic_kk"}, inplace=True)
        # Save the updated DataFrame to a CSV file
        output_dir = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}"
        os.makedirs(output_dir, exist_ok=True)
        self.data.rename(columns={"cluster": "max_topic"}, inplace=True)
        topic_map = self.data[['max_topic', 'topic_kk', 'year', 'CIK']]
        # Save topic_map as a CSV to /Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/embeddings_km10
        topic_map.to_csv(f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/{modelname}/topic_map_2006_2022.csv", index=False)

    
    @staticmethod
    def rebuild():
        years = range(2006, 2022 + 1)

        # Initialize empty lists to store embeddings, years, and CIKs
        all_embeddings = []
        all_years = []
        all_ciks = []

        # Loop through each year to load the embeddings and CIKs
        for year in years:
            # Load embeddings
            if year == 2012 or year == 2006:
                embeddings_file_path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{year}/embeddings{year}.pkl"
                embeddings = pd.read_pickle(embeddings_file_path)
                embeddings = np.array(embeddings)
                all_embeddings.append(embeddings)
            elif year > 2012:
                embeddings_file_path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{year}/embeddings{year}.npy"
                embeddings = np.load(embeddings_file_path)
                all_embeddings.append(embeddings)
            
            # Load CIKs
            pickle_file_path = f"/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A_files_cleaned/{year}/clean_texts_filter{year}.pkl"
            clean_texts_filter = pd.read_pickle(pickle_file_path)
            ciks = clean_texts_filter['cik']
            
            # Append embeddings, years, and CIKs to the lists
            
            all_years.append(np.full(ciks.shape[0], year))
            all_ciks.append(ciks)

        # Concatenate all embeddings, years, and CIKs along the first axis (rows)
        all_embeddings_concatenated = np.concatenate(all_embeddings, axis=0)
        all_years_concatenated = np.concatenate(all_years, axis=0)
        all_ciks_concatenated = np.concatenate(all_ciks, axis=0)

        # Create a DataFrame from the concatenated embeddings
        embedding_df = pd.DataFrame(all_embeddings_concatenated)

        # Add the year and CIK columns to the DataFrame
        embedding_df['year'] = all_years_concatenated
        embedding_df['CIK'] = all_ciks_concatenated
        # Save the DataFrame to a pickle file in the data folder, sister of the src folder:
        save_path = os.path.join('..', 'data', 'embeddings.pkl')
        embedding_df.to_pickle(save_path)
        print("Embeddings saved to data/embeddings.pkl")
        return embedding_df
    
    @staticmethod
    def from_pickle():
        # Load the embeddings DataFrame from the pickle file
        load_path = os.path.join('..', 'data', 'embeddings.pkl')
        embeddings = pd.read_pickle(load_path)
        print("Embeddings loaded from data/embeddings.pkl")
        return embeddings

    # Get embeddings for the words "innovation", "intellectual property", "patent", "knowledge capital", "clinical trial", and "software"
    inno_et = et.get_text_embeddings("innovation")
    ip_et = et.get_text_embeddings("intellectual property")
    patent_et = et.get_text_embeddings("patent")
    knowledge_capital_et = et.get_text_embeddings("knowledge capital")
    clinical_trial_et = et.get_text_embeddings("clinical trial")
    software_et = et.get_text_embeddings("software")

    # Stack embeddings into a matrix
    embeddings_matrix = np.vstack([inno_et, ip_et, patent_et, knowledge_capital_et, clinical_trial_et, software_et])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(embeddings_matrix)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(cosine_sim, dtype=bool))

    # Create a heatmap with the cosine similarity, masking the upper triangle
    words = ["innovation", "intellectual property", "patent", "knowledge capital", "clinical trial", "software"]
    sns.heatmap(cosine_sim, annot=True, xticklabels=words, yticklabels=words, cmap='viridis', mask=mask)
    plt.title('Cosine Similarity between Word Embeddings')
    plt.show()
    
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