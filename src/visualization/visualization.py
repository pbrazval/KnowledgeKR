# Data manipulation and numerical operations
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
# Date and time operations
from datetime import datetime

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_tile, geom_text, scale_fill_gradient2, labs, theme, element_text
import matplotlib.dates as mdates

# Utilities and performance optimization
import pyreadr
import time
import numba
import concurrent.futures
import risk_pricing as rp
import os

def announce_execution(func):
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__}...")
        result = func(*args, **kwargs)
        #print(f"{func.__name__} returned {result}")
        return result
    return wrapper

def label_topic_map(topic_map_unlabeled, name, cuts = [0, 0.2, 0.4, 0.6, 0.8, 1], **kwargs):
    topic_dict = {"dicfullmc10thr10defnob40noa1_4t": {"topic_0": "topic_kk", "topic_1": "topic_finl", "topic_2": "topic_sw", "topic_3": "topic_rawm"},
                  "dicfullmc10thr10defnob40noa0_8_4t": {"topic_0": "topic_0", "topic_1": "topic_1", "topic_2": "topic_2", "topic_3": "topic_kk"},
                  "dicfullmc10thr10defnob5noa1_4t": {"topic_0": "topic_0", "topic_1": "topic_1", "topic_2": "topic_2", "topic_3": "topic_kk"},
                  "dicfullmc10thr10defnob5noa0_8_4t": {"topic_0": "topic_energy", "topic_1": "topic_tech", "topic_2": "topic_finl", "topic_3": "topic_kk"},
                  "dicfullmc10thr10defnob40noa0_9_4t": {"topic_0": "topic_0", "topic_1": "topic_1", "topic_2": "topic_2", "topic_3": "topic_kk"},
                  "embeddings_km10_ipcs": {"topic_kk": "topic_kk"}}   
    
    if name[-3:] == "hdp":
        topic_map_labeled = topic_map_unlabeled.copy()
        return label_topic_map_hdp(topic_map_labeled, name, **kwargs)

    topic_map_labeled = topic_map_unlabeled.copy()
    if name in topic_dict.keys():
        topic_map_labeled.rename(columns = topic_dict[name], inplace=True)  
    # If name starts with "emb":
    
    topic_map_labeled['ntile_kk'] = (topic_map_labeled.
        groupby('year')['topic_kk'].
        transform(lambda x: pd.qcut(x, cuts, labels=False, duplicates='raise')))  
    
    return topic_map_labeled

def label_topic_map_hdp(topic_map_unlabeled, name, **kwargs):
    print("Labeling HDP topic map")
    if name == "dicfullmc10thr10defnob40noa0_8_hdp":
        topic_map_unlabeled['topic_kk'] = topic_map_unlabeled['topic_1'] + topic_map_unlabeled['topic_9']  + topic_map_unlabeled['topic_22'] + topic_map_unlabeled['topic_32']
        # Delete topic_1
        topic_map_labeled = topic_map_unlabeled.drop(columns = ["topic_1"])
        # Delete topic_9,...
        topic_map_labeled = topic_map_labeled.drop(columns = [f"topic_{i}" for i in range(9, 150)])

        cuts = [0, 0.85, 0.90, 0.95, 1]
        topic_map_labeled['ntile_kk'] = (topic_map_labeled.
                                        groupby('year')['topic_kk'].
                                        transform(lambda x: pd.qcut(x, cuts, labels=False, duplicates='raise')))  
        return topic_map_labeled


def label_dicfullmc10thr10defnob40noa1_4t(topic_map_unlabeled):
    k = 0
    labels = ["topic_" + str(k), str(k)]
    
    topic_map_labeled = topic_map_unlabeled.copy()
    
    topic_map_labeled.rename(columns={"topic_0": "topic_kk", "topic_1": "topic_finl", "topic_2": "topic_sw", "topic_3": "topic_rawm"}, inplace=True)
    
    return topic_map_labeled, labels

def personal_stargazer(mytable, textfolder, file_name, label, caption):
    latex_output = (mytable.
                to_latex(index=False, header=True, label = label, caption = caption, decimal = "."))
    file_name = textfolder + file_name
    with open(file_name, 'w') as f:
        f.write(latex_output)
    return None
# Data Manipulation using Pandas
def explore_topic_map(topic_map, figfolder, start_time, nt = 4):    

    fig_mean_tiy(topic_map, figfolder)
    
    tex_average_topic_loadings_by_high_tech(topic_map, figfolder, nt)

    tex_sample_topic_loadings(topic_map, figfolder)    

    skill_correlations = df_skill_correlations(topic_map)

    patent_correlations = df_patent_correlations(topic_map)
    
    fig_share_dominant_kk_by_ind(topic_map, figfolder)
    
    fig_heatmap_topicvskkpt(topic_map, figfolder, nt)

    fig_heatmap_topicvsikpt(topic_map, figfolder, nt)
    #print("running time:", time.time() - start_time)

    fig_histogram_kk(topic_map, figfolder)

    fig_histogram_kk_by_ind12(topic_map, figfolder)

    print("Finished!")
    return None

def explore_stoxwe(stoxwe, figfolder):
    plot_returns(stoxwe, figfolder)
    return None

def explore_stoxwe_with_pfs(stoxwe_with_pfs, figfolder):
    plot_returns(stoxwe_with_pfs, figfolder)
    return None

def explore_eret_we(eret_we, figfolder):
    eret_we_agg = (eret_we
            .groupby('yw')
            .mean())
    fig_h1b_vs_smb_kkhml(eret_we_agg, figfolder)
    tex_HKR_vs_mktrf(eret_we_agg, figfolder)
    tex_summary_statistics(eret_we_agg, figfolder)
    return None

@announce_execution
def tex_HKR_vs_mktrf(eret_we_agg, figfolder):
    summary = rp.HKR_vs_mktrf(eret_we_agg)
    # Define the file path for the tex file
    tex_content = summary.as_latex()
    
    # Remove the footnotes section from the LaTeX content
    tex_content = tex_content.split("\\bigskip")[0]
    
    # Define the file path for the tex file
    tex_file_path = figfolder + "HKR_vs_mktrf.tex"
    
    # Write the modified summary to the tex file
    with open(tex_file_path, 'w') as tex_file:
        tex_file.write(tex_content)

@announce_execution
def fig_h1b_vs_smb_kkhml(eret_we_agg, figfolder):
    plt.figure()
    factors = (eret_we_agg
              .loc[:, ['SMB', 'HKR']]
             )
    factors = factors + 1
    factors = factors.cumprod()
    h1b_date = pd.to_datetime("20200623", errors="coerce", format="%Y%m%d")
    year_week = h1b_date.strftime('%Y%U')
    factors = factors.mul(100 / factors.loc[int(year_week)])
    factors.index  = factors.index // 100 + (factors.index % 100) / 53
    # Include a vertical line at 2020
    factors.loc[2015:2021].plot()
    plt.axvline(2020 + 1/6, color='r', linestyle='--')
    # Include a blue vertical line at the date of the H1B suspension: 
    plt.axvline(2020 + 26/53, color='b', linestyle='--')
    plt.xlabel("Year")
    plt.ylabel("Cumulative return (Normalize to 100 at June 2020)")
    plt.title("Cumulative return of SMB and HKR factors")
    plt.savefig(figfolder + "h1b_vs_smb_kkhml.jpg", dpi=600)
    plt.close()

@announce_execution
def tex_summary_statistics(eret_we_agg, figfolder):
    # Pick columns only in set {'MktRF', 'SMB', 'HML', 'HKR', 'RMW', 'CMA'}:
    desired_columns = {'Mkt.RF', 'SMB', 'HML', 'HKR', 'RMW', 'CMA', 'HKR_SB'}

    eret_we_agg = eret_we_agg.loc[:, [col for col in eret_we_agg.columns if col in desired_columns]]
        # Convert values from weekly to monthly:
    summary = eret_we_agg.describe()
    # Transpose the dataframe:
    summary = summary.T
    # Multiply the values by 4.35 of all the columns except count:
    summary.loc[:, summary.columns != 'count'] = summary.loc[:, summary.columns != 'count'] * 4.35
    summary['Sharpe'] = summary['mean'] / summary['std']

    summary[['mean', 'std', 'min', 'max', '25%', '50%', '75%']] =\
        summary[['mean', 'std', 'min', 'max', '25%', '50%', '75%']].applymap(to_percentage)

    summary = summary[['count', 'mean', 'std', 'Sharpe', 'min', '25%', '50%', '75%', 'max']]
    summary['Sharpe'] = summary['Sharpe'].round(3)
    print("Now with Sharpe ratio")
    
    save_table_dual(figfolder, summary, "summary_statistics")

def to_percentage(x):
    return f"{x * 100:.2f}%"

def save_table_dual(figfolder, table, filename):
    with open(figfolder + filename + ".tex", 'w') as tex_file:
        tex_file.write(table.to_latex(index = False, header = True))

    # Print as HTML as well:
    with open(figfolder + filename + ".html", 'w') as html_file:
        html_file.write(table.to_html())

@announce_execution
def explore_fmb(fmb_list, figfolder):
    tex_summary(fmb_list, figfolder)

def explore_stoxda(stoxda, cequity_mapper, topic_map, figfolder):
    # Plot the Amazon stock prices
    # amazon_graph(stoxda, figfolder)
    # Format date and create 'y' and 'ym' columns
    stoxda['date'] = pd.to_datetime(stoxda['date'])
    stoxda['y'] = stoxda['date'].dt.year
    stoxda['ym'] = stoxda['y']*100 + stoxda['date'].dt.month
    stox = pd.merge(stoxda, cequity_mapper, left_on=['PERMNO', 'y'], right_on=['PERMNO', 'year'], how='left')
    stox = stox[stox['crit_ALL'] == 1]
    stox = pd.merge(stox, topic_map, left_on=['PERMNO', 'y'], right_on=['LPERMNO', 'year'], how='left')
    topic_map = topic_map.assign(me=lambda x: x['csho'] * x['prcc_f']) #groupby(['PERMNO', hue_var])
    stox.set_index('date', inplace=True)

    plot_moment(stoxda, cequity_mapper, topic_map, figfolder, "kurtosis", "Y")
    plot_moment(stoxda, cequity_mapper, topic_map, figfolder, "skewness", "Y")
    #plot_moment(stoxda, cequity_mapper, topic_map, figfolder, "kurtosis", "M")
    #plot_moment(stoxda, cequity_mapper, topic_map, figfolder, "skewness", "M")
    return None

from stargazer.stargazer import Stargazer

@announce_execution
def tex_summary(fmb_list, figfolder):
    stargazer = Stargazer(fmb_list)
    stargazer.significant_digits(5)
    stargazer.title('Fama-MacBeth Regressions')
    stargazer.covariate_order(['MktRF', 'SMB', 'HML', 'HKR', 'RMW', 'CMA'])
    stargazer.show_degrees_of_freedom(False)
    stargazer.show_f_statistic = False
    stargazer.show_residual_std_err = False
    # stargazer.show_footer = False
    stargazer.dep_var_name = "Dep. var: Portfolio weekly excess return - "
    # Create a vector with "model_1", "model_2", with the length of fmb_list:
    model_vector = [f"model_{i+1}" for i in range(len(fmb_list))]
    stargazer.custom_columns(model_vector)
    result = stargazer.render_latex()
    # Save to the right place:
    with open(os.path.join(figfolder, "fmb_results.tex"), "w") as text_file:
        text_file.write(result)

    result = stargazer.render_html()
    file_path = os.path.join(figfolder, "fmb_results.html")
    with open(file_path, "w") as text_file:
        text_file.write(result)

@announce_execution
def fig_histogram_kk_by_ind12(topic_map, figfolder):
    plt.figure()
    unique_values = topic_map['ind12'].unique()  # Get unique values from 'ind12' column

    # Set up the figure size and the number of subplots (3 rows of 4 histograms each)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration

    for i, value in enumerate(unique_values):
        # Filter data for each category of 'ind12'
        data_filtered = topic_map[topic_map['ind12'] == value]
        
        # Plot histogram on the respective subplot
        sns.histplot(data=data_filtered, x='topic_kk', bins=15, kde=False, ax=axes[i], stat='probability')
        axes[i].set_title(f'{value}', fontsize=18)
        axes[i].set_xlabel('topic_kk')
        axes[i].set_ylabel('Probability')
        axes[i].set_xlim(0.05, 0.45)  # Set x-axis limits for consistency

    
    plt.tight_layout()
    # Save the figure before displaying it
    plt.savefig(figfolder + "hist_kk_by_ind12.png", dpi=600)
    plt.close()

@announce_execution
def fig_histogram_kk(topic_map, figfolder):
    plt.figure()
    sns.histplot(data=topic_map, x='topic_kk', bins=20, kde=False, stat='probability')  # Adjust bins as needed
    plt.title('Histogram of topic_kk')
    plt.xlabel('Values of topic_kk')
    plt.ylabel('Probability')
    # Save the figure before showing it
    plt.savefig(figfolder + "hist_kk_agg.png", dpi=600)
    plt.close()

@announce_execution
def fig_heatmap_topicvsikpt(topic_map, figfolder, nt):
    topic_map_positiveikpt = topic_map.dropna(subset = ["K_int_Know", "at", "K_int"]).loc[(topic_map["K_int"] > 0), :]
    topic_map_positiveikpt['ikpt_ntile'] = topic_map_positiveikpt.groupby('year')["K_int"].transform(lambda x: pd.qcut(x, nt, labels=False, duplicates='drop'))

    firms_by_ik = (topic_map_positiveikpt
                .groupby(['ntile_kk', 'ikpt_ntile'])
                .size()
                .reset_index(name='count'))

    plot = (ggplot(firms_by_ik, aes(x='ntile_kk', y='ikpt_ntile', fill='count')) +
            geom_tile(aes(fill='count')) +  # Use geom_tile for heatmap squares
            geom_text(aes(label='round(count, 2)'), size=20) +  # Add text labels
            scale_fill_gradient2(low="white", high="red", mid="pink", midpoint=firms_by_ik['count'].mean()) +  # Gradient fill
            labs(x="N-tiles of Intangible Capital Risk measured by Topic_kk",
                y="N-tiles of Intangible Capital Intensity",
                fill='Count') +  # Labels
            theme(legend_title=element_text(size=14),  # Adjust legend title font size
                legend_text=element_text(size=14),  # Adjust legend text font size
                axis_title=element_text(size=14),  # Adjust axis titles font size
                axis_text=element_text(size=14))  # Adjust axis texts font size
        )


    plot.save(figfolder + "topicvsikpt_hm.png", dpi=600, width=10, height=8, verbose=False)

@announce_execution
def fig_heatmap_topicvskkpt(topic_map, figfolder, nt):
    topic_map_positivekkpt = (topic_map
                              .dropna(subset = ["K_int_Know", "at", "K_int"])
                              .loc[(topic_map["K_int_Know"] > 0), :])
    topic_map_positivekkpt['kkpt_ntile'] = topic_map_positivekkpt.groupby('year')["K_int_Know"].transform(lambda x: pd.qcut(x, nt, labels=False, duplicates='drop'))
    
    firms_by_kk = (topic_map_positivekkpt
                .groupby(['ntile_kk', 'kkpt_ntile'])
                .size()
                .reset_index(name='count'))

    plot = (ggplot(firms_by_kk, aes(x='ntile_kk', y='kkpt_ntile', fill='count')) +
            geom_tile(aes(fill='count')) +  # Use geom_tile for heatmap squares
            geom_text(aes(label='round(count, 2)'), size=20) +  # Add text labels
            scale_fill_gradient2(low="white", high="red", mid="pink", midpoint=firms_by_kk['count'].mean()) +  # Gradient fill
            labs(x="N-tiles of Knowledge Capital Risk measured by Topic_kk",
                y="N-tiles of Knowledge Capital Intensity",
                fill='Count') +  # Labels
            theme(legend_title=element_text(size=14),  # Adjust legend title font size
                legend_text=element_text(size=14),  # Adjust legend text font size
                axis_title=element_text(size=14),  # Adjust axis titles font size
                axis_text=element_text(size=14))  # Adjust axis texts font size
        )
    
    # Save the plot
    plot.save(figfolder + "topicvskkpt_hm.png", dpi=600, width=10, height=8, verbose=False)

@announce_execution
def df_patent_correlations(topic_map):
    patentcor = (topic_map.
                filter([col for col in topic_map.columns if col.startswith("topic")]+["xir_cumsum"]))
    patent_correlations = patentcor.corr(method = "spearman").loc["xir_cumsum", [col for col in patentcor.columns if col.startswith("topic")]]
    return patent_correlations

@announce_execution
def df_skill_correlations(topic_map):
    skillcor = (topic_map.
                filter([col for col in topic_map.columns if col.startswith("topic")]+["Skill"]))
    # Create an array with the correlation of each column that starts with "topic" with column "Skill":
    skill_correlations = skillcor.corr(method = "spearman").loc["Skill", [col for col in skillcor.columns if col.startswith("topic")]]
    return skill_correlations

@announce_execution
def fig_share_dominant_kk_by_ind(topic_map, figfolder):
    maxkk = max(topic_map["ntile_kk"])
    firms_by_ind = (topic_map
                    .loc[topic_map["ntile_kk"]==maxkk]
                    .groupby(["year", "ind12"])
                    .agg(count = ("year", "size"),
                        totalat = ("at", "sum"))
                    .reset_index(inplace = False))
    # Convert year and ind12 from index to columns:
    
    stackedplot_n = sns.barplot(data=firms_by_ind, x='year', y='count', hue='ind12', dodge=False)
    stackedplot_n.set_ylabel('Share of all dominant-KK firms')
    stackedplot_n.set_xlabel('Year')
    stackedplot_n.set_title('Share of all dominant-KK firms by Industry')
    stackedplot_n.legend(title='Industry', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)

    # Save plot above to "stackedplot_at.png" inside the "figfolder" directory:
    stackedplot_n.figure.savefig(figfolder + "stackedplot_n.png", bbox_inches='tight', dpi=300)
    plt.close()

@announce_execution
def tex_sample_topic_loadings(topic_map, figfolder):
    np.random.seed(139)
    sample_topics = (topic_map.
                    filter([col for col in topic_map.columns if col.startswith("topic")]+["conm", "year"]).
                    sample(n = 10).
                    sort_values(by = "conm").
                    rename(columns = {"conm": "Company_Name"}).
                    apply(lambda x: round(x, 3) if x.name.startswith('topic') else x, axis = 0))

    personal_stargazer(sample_topics, figfolder, "sample_topics", "tab:sample_topics", "Sample of Topic Loadings")

@announce_execution
def tex_average_topic_loadings_by_high_tech(topic_map, figfolder, nt):

    bytech = topic_map.dropna(subset = "hi_tech")
            
    bytech = (bytech.
            filter([col for col in bytech.columns if col.startswith("topic")]+["hi_tech"]).
            groupby("hi_tech").
            agg(lambda x: round(x.mean(), 3) if x.name.startswith('topic') else x))

    personal_stargazer(bytech, figfolder, "bytech", "tab:bytech", "Average Topic Loadings by High Tech")
    return None

@announce_execution
def amazon_graph(amazon_nov01_short, figfolder):
    # Convert 'Date' column to datetime format
    amazon_nov01_short['Date'] = pd.to_datetime(amazon_nov01_short['Date'])
    
    # Find the specific 'nasdaq' and 'amazon' values for November 13, 2001
    specific_date_values = amazon_nov01_short[amazon_nov01_short['Date'] == pd.to_datetime("2001-11-13")][['nasdaq', 'amazon']]
    
    # Index 'nasdaq' and 'amazon' to 100 based on their values on November 13, 2001
    amazon_nov01_short['nasdaq'] = 100 * amazon_nov01_short['nasdaq'] / specific_date_values['nasdaq'].values[0]
    amazon_nov01_short['amazon'] = 100 * amazon_nov01_short['amazon'] / specific_date_values['amazon'].values[0]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot('Date', 'nasdaq', data=amazon_nov01_short, color='blue', label='NASDAQ')
    plt.plot('Date', 'amazon', data=amazon_nov01_short, color='red', label='Amazon')
    plt.axvline(x=pd.to_datetime("2001-11-13"), linestyle='--', color='black')
    
    # Setting the title, labels, legend, and formatting the x-axis dates
    plt.title("NASDAQ vs Amazon Stock Prices in November 2001 (11/13/01 = 100)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.grid(True)
    
    # Save the plot
    plt.savefig(figfolder + "amazon_nov01.png", dpi=600)
    plt.close()

@announce_execution
def fig_mean_tiy(df, figfolder):
    # Select columns 'year' and those starting with 'topic_'
    df_filtered = df.filter(regex='^year|topic_.*')
    
    # Calculate the average topic intensity for each year
    avg_df = df_filtered.groupby('year').mean().reset_index()
    
    # Reshape the DataFrame from wide to long format
    long_df = pd.melt(avg_df, id_vars=['year'], var_name='topic', value_name='intensity')
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=long_df, x='year', y='intensity', hue='topic')
    
    # Setting labels and title
    plt.xlabel("Year")
    plt.ylabel("Topic Intensity")
    plt.title("Mean Topic Intensity by Year")
    plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust font sizes for readability
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, title_fontsize='13')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(figfolder + "mean_tiy.jpg", dpi=600)
    plt.close()
    return None

@announce_execution
def filecounter(textfolder):
    path = "/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/1A files/"
    year_file_counts = {}
    for year in range(2006, 2023):
        folder_path = os.path.join(path, str(year))
        if os.path.exists(folder_path):
            subfolder_names = ["Q1", "Q2", "Q3", "Q4"]
            year_file_count = 0

            for subfolder in subfolder_names:
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.exists(subfolder_path):
                    year_file_count += len(os.listdir(subfolder_path))
            
            year_file_counts[year] = year_file_count

    # Load the CSV file
    lemmat_counts = pd.read_csv("/Users/pedrovallocci/Documents/PhD (local)/Research/By Topic/Measuring knowledge capital risk/output/descriptive/lemmat_counts.csv")
    # Create a DataFrame from year_file_counts
    file_counts_df = pd.DataFrame(list(year_file_counts.items()), columns=['Year', 'Total_1As'])
    # Merge or update lemmat_counts with the new file counts
    lemmat_counts = pd.merge(lemmat_counts, file_counts_df, on='Year', how='left')
    lemmat_counts.rename(columns={'Count': 'Filtered'}, inplace=True)
    lemmat_counts = lemmat_counts[['Year', 'Total_1As', 'Filtered']]
    lemmat_counts = lemmat_counts.rename(columns = {"Year": "Year", "Total_1As": "Total", "Filtered": "Filtered"})

    # Generate LaTeX table (manually or using a library like pylatex)
    # For simplicity, we're saving the DataFrame to a CSV file
    # You can manually convert this CSV to a LaTeX table or use Python libraries if needed
    #output_filepath = os.path.join(textfolder, "file_counts.tex")

    # Save to a .tex file
    #lemmat_counts.to_latex(output_filepath, index=False)
    save_table_dual(textfolder, lemmat_counts, "file_counts")

@announce_execution
def tex_compare_kk_measures(comparison_measures, figfolder):
    """
    Generate a LaTeX table from comparison_measures DataFrame and save it.

    Parameters:
    comparison_measures (pd.DataFrame): DataFrame containing the data to be tabled.
    figfolder (str): Folder path to save the LaTeX file.
    """
    # Convert DataFrame to LaTeX
    save_table_dual(figfolder, comparison_measures, "corr_measures")

    # latex_content = comparison_measures.to_latex(index=False, header=True)

    # # Additional LaTeX table customizations can be done here by manipulating latex_content string
    
    # # Define file path
    # file_path = f"{figfolder}/corr_measures.tex"
    
    # # Save LaTeX table to file
    # with open(file_path, "w") as latex_file:
    #     latex_file.write(latex_content)
    # return None

@announce_execution
def plot_moment(stox, figfolder, moment_name, frequency = 'Y', hue_var = "ntile_kk", asset_weighted = False):
    if moment_name == 'kurtosis':
        statfunc = pd.Series.kurt
    elif moment_name == 'skewness':
        statfunc = pd.Series.skew
    else:
        raise ValueError("Invalid moment_name. Choose 'kurtosis' or 'skewness'.")

    moment_df = stox.groupby(['PERMNO', hue_var]).resample(frequency)['RET'].apply(statfunc)
    moment_df = moment_df.reset_index()
    if not asset_weighted:
        aw_suffix = "asset-weighted"
        aw_suffix_short = "aw"
        stox_by_kk = (moment_df.reset_index().groupby([hue_var, 'date'])
                        .agg(avg_kurt=('RET', 'mean'))
                        .reset_index())
        #moment_df= moment_df.reset_index()
    else:
        aw_suffix = "not asset-weighted"
        aw_suffix_short = "naw"
        stoxme_df = stox.reset_index()[['PERMNO', 'date', 'me']]
        moment_df = moment_df.merge(stoxme_df, on=['PERMNO', 'date'], how='left')
        moment_df = moment_df.dropna(subset=['RET', 'me'])
        stox_by_kk = (moment_df.groupby([hue_var, 'date'])
                            .apply(lambda x: pd.Series({'avg_kurt': (x['RET'] * x['me']).sum() / x['me'].sum()}))
                            .reset_index())

    large_palette = sns.color_palette('husl', 8)
    # Find the two lowest and two highest values of ntile_kk:
    ntile_kk_values = stox_by_kk[hue_var].unique()
    sorted_values = sorted(ntile_kk_values)

    lowest_two = sorted_values[:2]
    highest_two = sorted_values[-2:]
    keep = lowest_two + highest_two

    # Filter stox_by_kk to only include ntile_kk values equal to the two lowest and two highest values of ntile_kk
    if hue_var == 'ntile_kk':
        stox_by_kk = stox_by_kk[stox_by_kk[hue_var].isin(keep)]
    if hue_var == 'max_topic':
        # Filter only rows where max_topic is between 0 and 7 (inclusive):
        stox_by_kk = stox_by_kk[stox_by_kk[hue_var].between(0, 7)]
    sns.lineplot(data=stox_by_kk, x='date', y='avg_kurt', hue=hue_var, palette=large_palette)
    plt.title(f"{moment_name} ({aw_suffix}) over time by {hue_var}")
    plt.xlabel("Year-Month")
    plt.ylabel(f"{moment_name}")
    plt.legend(title='Group', fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{figfolder}/{moment_name}_by_{hue_var}_{frequency}_{aw_suffix_short}.jpg", dpi=600)
    plt.clf()
    return None

@announce_execution
def plot_returns(stoxwe_with_pfs, figfolder):
    # Need to add:
    # Asset-weighted weekly returns
    # Asset-weighted weekly returns, 3MA
    # Asset-weighted accumulated weekly returns
    # Weekly standard deviation of returns by n-tile, four-week MA
    # Kurtosis over time by group
    def week_to_date(week):
        week = str(int(week))
        year = int(week[:4])
        week = int(week[4:6])
        return datetime.strptime(f'{year}-{week}-1', "%Y-%W-%w")
    
    # stoxwe_with_pfs.yw = stoxwe_add.yw.astype(str)
    # stoxwe_with_pfs.yw = stoxwe_with_pfs.yw.apply(week_to_date)
    if isinstance(stoxwe_with_pfs['yw'][0], (int, float)):
        stoxwe_with_pfs['yw'] = pd.to_numeric(stoxwe_with_pfs['yw'], errors='coerce')
    # Apply the function to the 'yw' column
        stoxwe_with_pfs['yw'] = stoxwe_with_pfs['yw'].apply(lambda x: week_to_date(x) if not pd.isna(x) else pd.NaT)

    we_ret_bybin = (stoxwe_with_pfs
                    .groupby(['yw', 'ntile_kk'])
                    .agg(
                        eret=('eretw', lambda x: np.sum(x * stoxwe_with_pfs.loc[x.index, 'me']) / np.sum(stoxwe_with_pfs.loc[x.index, 'me'])),
                        sderet=('eretw', 'std')
                        )
    .reset_index()
    .groupby('ntile_kk')
    .apply(lambda x: x.assign(eret_accum=x['eret'].cumsum()))
    .reset_index(drop=True))
    

    qt_ret_bygroup = (we_ret_bybin
                      .groupby('ntile_kk')
                      .apply(lambda x: x.assign(eret3ma=x['eret'].rolling(window=13, min_periods=1).mean()))
                      .reset_index(drop=True))
    
    plt.figure()
    sns.lineplot(data=we_ret_bybin, x='yw', y='eret', hue='ntile_kk')
    plt.xlabel("Year-month")
    plt.ylabel("Asset-weighted weekly returns")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(figfolder + "aw_wr.jpg", dpi=600)
    plt.clf()
    
    plt.figure()
    sns.lineplot(data=qt_ret_bygroup, x='yw', y='eret3ma', hue='ntile_kk')
    plt.xlabel("Year-week")
    plt.ylabel("Asset-weighted weekly returns, 3MA")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(figfolder + "aw_wr_3ma.jpg", dpi=600)
    plt.clf()
    
    plt.figure()
    sns.lineplot(data=we_ret_bybin, x='yw', y='eret_accum', hue='ntile_kk')
    plt.xlabel("Year-week")
    plt.ylabel("Asset-weighted accumulated weekly returns")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(figfolder + "aw_accum_wr.jpg", dpi=600)
    plt.clf()

    plt.figure()
    sns.lineplot(data=we_ret_bybin, x='yw', y='sderet', hue='ntile_kk')
    plt.xlabel("Year-week")
    plt.ylabel("Weekly standard deviation of returns by n-tile")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0.02,0.14)
    plt.savefig(figfolder + "sd_eret.jpg", dpi=600)
    plt.clf()
    
    return None    