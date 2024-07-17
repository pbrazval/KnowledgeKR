# Data manipulation and numerical operations
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, ttest_1samp
# Date and time operations
from datetime import datetime

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_tile, geom_text, scale_fill_gradient2, labs, theme, element_text
import matplotlib.dates as mdates

# Utilities and performance optimization
import risk_pricing as rp
import os
from .latex_utils import *

def announce_execution(func):
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__}...")
        result = func(*args, **kwargs)
        #print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@announce_execution
def label_topic_map(topic_map_unlabeled, name, cuts = [0, 0.2, 0.4, 0.6, 0.8, 1], **kwargs):
    topic_dict = {"dicfullmc10thr10defnob40noa1_4t": {"topic_0": "KKR", "topic_1": "topic_finl", "topic_2": "topic_sw", "topic_3": "topic_rawm"},
                  "dicfullmc10thr10defnob40noa0_8_4t": {"topic_0": "topic_0", "topic_1": "topic_1", "topic_2": "topic_2", "topic_3": "KKR"},
                  "dicfullmc10thr10defnob5noa1_4t": {"topic_0": "topic_0", "topic_1": "topic_1", "topic_2": "topic_2", "topic_3": "KKR"},
                  "dicfullmc10thr10defnob5noa0_8_4t": {"topic_0": "topic_energy", "topic_1": "topic_tech", "topic_2": "topic_finl", "topic_3": "KKR"},
                  "dicfullmc10thr10defnob40noa0_9_4t": {"topic_0": "topic_0", "topic_1": "topic_1", "topic_2": "topic_2", "topic_3": "KKR"},
                  "embeddings_km10_ipcs": {"KKR": "KKR"}}   
    
    if name == "dicfullmc10thr10defnob40noa0_8_4t":
        cuts = [0, 0.85, 0.90, 0.95, 1]
    elif name == "dicfullmc10thr10defnob40noa0_8_hdp":
        cuts = [0, 0.85, 0.90, 0.95, 1]

    if name[-3:] == "hdp":
        topic_map_labeled = topic_map_unlabeled.copy()
        topic_map_labeled = label_topic_map_hdp(topic_map_labeled, name, **kwargs)
        return topic_map_labeled

    topic_map_labeled = topic_map_unlabeled.copy()
    if name in topic_dict.keys():
        topic_map_labeled.rename(columns = topic_dict[name], inplace=True)  
    # If name starts with "emb":
    if name[:3] == "emb":
        # Rename column "topic_kk" to "KKR":
        topic_map_labeled.rename(columns = {"topic_kk": "KKR"}, inplace=True)

    topic_map_labeled['ntile_kk'] = (topic_map_labeled.
        groupby('year')['KKR'].
        transform(lambda x: pd.qcut(x, cuts, labels=False, duplicates='raise')))  
    topic_map_labeled['ntile_kk'] = topic_map_labeled['ntile_kk'] + 1
    topic_map_labeled['ntile_kk'] = topic_map_labeled['ntile_kk'].astype(int)

    return topic_map_labeled

def label_topic_map_hdp(topic_map_unlabeled, name, **kwargs):
    print("Labeling HDP topic map")
    if name == "dicfullmc10thr10defnob40noa0_8_hdp":
        topic_map_unlabeled['KKR'] = topic_map_unlabeled['topic_1'] + topic_map_unlabeled['topic_9']  + topic_map_unlabeled['topic_22'] + topic_map_unlabeled['topic_32']
        # Delete topic_1
        topic_map_labeled = topic_map_unlabeled.drop(columns = ["topic_1"])
        # Delete topic_9,...
        topic_map_labeled = topic_map_labeled.drop(columns = [f"topic_{i}" for i in range(9, 150)])

        cuts = [0, 0.85, 0.90, 0.95, 1]
        topic_map_labeled['ntile_kk'] = (topic_map_labeled.
                                        groupby('year')['KKR'].
                                        transform(lambda x: pd.qcut(x, cuts, labels=False, duplicates='raise')))  
        return topic_map_labeled

# Data Manipulation using Pandas
@announce_execution
def explore_topic_map(topic_map, figfolder, nt = 4):    

    topic_map['kkpt_intensity'] = topic_map['K_int_Know'] / topic_map['at']
    tex_calculate_correlation_matrix(figfolder, topic_map)

    tex_descriptive_statistics(figfolder, topic_map)    

    fig_mean_tiy(topic_map, figfolder)
    
    tex_average_topic_loadings_by_high_tech(topic_map, figfolder, nt)

    tex_sample_topic_loadings(topic_map, figfolder)    

    skill_correlations = df_skill_correlations(topic_map)

    patent_correlations = df_patent_correlations(topic_map)
    
    fig_share_dominant_kk_by_ind(topic_map, figfolder)
    
    fig_heatmap_topicvskkpt(topic_map, figfolder, nt)

    fig_heatmap_topicvsikpt(topic_map, figfolder, nt)

    fig_histogram_kk(topic_map, figfolder)

    fig_histogram_kk_by_ind12(topic_map, figfolder)

    print("Finished!")
    return None

@announce_execution
def tex_descriptive_statistics(figfolder, topic_map):
    # Create subset of topic_map
    topic_map_subset = topic_map[['KKR', 'Skill', 'xir_cumsum', 'K_int_Know', 'K_int', 'at', 'ppegt', 'kkpt_intensity']]
    
    # Rename columns
    topic_map_subset.columns = ['KKR', 'Skill', 'Patent Intensity', 'Knowledge Cap. (PT)', 'Intangible Cap. (PT)', 'Total Assets', 'Gross PPE', 'Knowledge Cap. (PT) Intensity']
    
    # Reorder columns
    topic_map_subset = topic_map_subset[['Total Assets', 'Gross PPE', 'Intangible Cap. (PT)', 'Knowledge Cap. (PT)', 'Knowledge Cap. (PT) Intensity', 'KKR', 'Skill', 'Patent Intensity']]
    
    # Calculate descriptive statistics
    desc = topic_map_subset.describe()
    
    # Round to specified decimal places
    desc = desc.round({
        'Total Assets': 1,
        'Gross Property, Plant and Equipment': 0,
        'Intangible Capital (PT)': 0,
        'Knowledge Capital (PT)': 0,
        'Knowledge Capital (PT) Intensity': 3,
        'KKR': 3,
        'Skill': 3,
        'Patent Intensity': 3
    })
    
    # Function to remove trailing zeros
    def remove_trailing_zeros(x):
        return '{:g}'.format(float(x))
    
    # Apply the function to remove trailing zeros
    desc_no_zeros = desc.applymap(remove_trailing_zeros)
    
    # Transpose the result
    result = desc_no_zeros.T
    # Rename columns:
    result = result.rename(columns = {"count": "Count", "mean": "Mean", "std": "SD", "min": "Min", "25%": "25\\%", "50%": "50\\%", "75%": "75\\%", "max": "Max"})

    save_table_dual(figfolder, result, "descriptive_statistics", row_names = True, tabular = True)
    
    return result

@announce_execution
def tex_calculate_correlation_matrix(figfolder, topic_map):
    topic_map_subset = topic_map[['kkpt_intensity', 'xir_cumsum', 'Skill', 'KKR']]
    topic_map_subset.columns = ['KK Int.', 'Pat.Int.', 'Skill', 'KKR']
    corr = topic_map_subset.corr(method = 'spearman')
    corr = corr.round(3)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Apply the mask to the correlation matrix
    corr_lower = corr.where(~mask)
    
    # Print only 3 digits after the decimal point:
    corr_lower = corr_lower.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else '')
    # Delete the first row and the last column:
    corr_lower = corr_lower.iloc[1:, :-1]
    save_table_dual(figfolder, corr_lower, "correlation_measures", row_names=True, tabular = True)

@announce_execution
def explore_stoxwe(stoxwe, figfolder):
    plot_returns(stoxwe, figfolder)
    return None

@announce_execution
def explore_stoxwe_with_pfs(stoxwe_with_pfs, figfolder):
    plot_returns(stoxwe_with_pfs, figfolder)
    return None

def explore_eret_we(eret_we5, eret_we_pct5, figfolder, log_returns = False):
    svar_qwe = create_svar()
    eret_we_agg = preprocess_eret_we(eret_we5)
    eret_we_agg = eret_we_agg.merge(svar_qwe, on='yw', how='left')
    eret_qwe_agg = rp.pseudo_monthly(eret_we_agg) 

    fig_h1b_vs_smb_kkhml(eret_we_agg, figfolder)
    tex_HKR_vs_mktrf(eret_qwe_agg, figfolder)
    tex_fmb_results_statistics(eret_we_agg, figfolder)
    tex_gmm_results(eret_we_pct5, None, figfolder)
    return None

def tex_gmm_results(eret_we3, eret_we5, figfolder, case = 3):
    summary1 = rp.gmm(eret_we3, factors = ['Mkt.RF', 'HKR'], formula = "HKR + Mkt.RF")
    summary3 = rp.gmm(eret_we3, factors = ['Mkt.RF', 'SMB', 'HML', 'HKR'], formula = "HKR + Mkt.RF + SMB + HML")
    if case == 5:
        summary5 = rp.gmm(eret_we5, factors = ['Mkt.RF', 'SMB', 'HML', 'HKR', 'RMW', 'CMA'], formula = "HKR + Mkt.RF + SMB + HML + RMW + CMA")
        covariate_order = ['HKR', 'Mkt.RF', 'SMB', 'HML', 'RMW', 'CMA']
        gmmlist = [summary1, summary3, summary5]
        model_vector = ["Mkt", "FF3", "FF5"]
    else:
        covariate_order = ['HKR', 'Mkt.RF', 'SMB', 'HML']
        gmmlist = [summary1, summary3]
        model_vector = ["Mkt", "FF3"]    
    export_gmm_results(figfolder, covariate_order, gmmlist, model_vector)

def export_gmm_results(figfolder, covariate_order, gmmlist, model_vector):
    stargazer = Stargazer(gmmlist)
    stargazer.covariate_order(covariate_order)
    stargazer.show_degrees_of_freedom(False)
    stargazer.show_f_statistic = False
    stargazer.show_residual_std_err = False
    stargazer.table_label = "tab:gmm"
    stargazer.custom_columns(model_vector)
    result = stargazer.render_latex()
    # Save to the right place:
    result = extract_tabular_content(result)
    with open(os.path.join(figfolder, "gmm.tex"), "w") as text_file:
        text_file.write(result)

    result = stargazer.render_html()
    file_path = os.path.join(figfolder, "gmm.html")
    with open(file_path, "w") as text_file:
        text_file.write(result)

def preprocess_eret_we(eret_we):
    eret_we_agg = (eret_we
            .groupby('yw')
            .mean())
    eret_we_agg = eret_we_agg.reset_index()
    eret_we_agg.yw = eret_we_agg.yw.astype(int)
    # Apply the function to the yw column
    # eret_we_agg['date'] = convert_yw_to_date(eret_we_agg['yw'])
    return eret_we_agg

def convert_yw_to_date(yw):
        year = yw // 100
        week = yw % 100
        # Create a date corresponding to the first day of the year
        date = pd.to_datetime(year.astype(str) + '0101', format='%Y%m%d')
        # Find the first Sunday of the year for each date
        first_sunday = date.apply(lambda x: x + pd.to_timedelta((6 - x.weekday()) % 7, unit='D'))
        # Add the number of weeks to get the Sunday of the specified week
        return first_sunday + pd.to_timedelta((week - 1) * 7, unit='D')

def create_svar():
    file_path = '/Users/pedrovallocci/Documents/PhD (local)/Research/Github/KnowledgeKRisk_10Ks/data/ff3fd.csv'
    df = pd.read_csv(file_path, skiprows=1, header=None)

# # Manually set the correct column names
    df.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF']

# # Convert the date column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    # Convert date to year-week format:

    # Merton (1980) requires log-returns here.
    df['Mkt-RF'] = np.log(1 + df['Mkt-RF'] / 100)

# # Resample the data by week and calculate the sum of squares of log returns for each week
    # min_date = eret_qwe_agg['date'].min()
    # myrange = df['date'] >= min_date - pd.DateOffset(weeks=1)
    svar_qwe = df.resample('W', on='date')['Mkt-RF'].apply(lambda x: np.sum(x**2))

    svar_qwe = svar_qwe.reset_index()
    # Convert date to year-week format:
    svar_qwe.columns = ['date', 'SVAR']
    svar_qwe['yw'] = svar_qwe['date'].dt.strftime('%Y%U').astype(int)
# Filter svar_qwe with date > 2006-07-23
    return svar_qwe

@announce_execution
def tex_HKR_vs_mktrf(eret_qwe_agg, figfolder):
    print("Now using pseudo-monthly returns")
    rp.HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, periods = [1, 3, 13, 26, 39, 52, 65], moment = 1, regressors = "SMB + HML + HKR", skip_crises=False)
    rp.HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, periods = [1, 3, 13, 26, 39, 52, 65], moment = 1, regressors = "HKR", skip_crises=False)
    rp.HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, periods = [1, 3, 13, 26, 39, 52, 65], moment = 1, regressors = "SMB + HML + HKR", skip_crises=True)
    rp.HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, periods = [1, 3, 13, 26, 39, 52, 65], moment = 1, regressors = "HKR", skip_crises=True)
    rp.HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, periods = [1, 3, 13, 26, 39, 52, 65], moment = 2, regressors = "SMB + HML + HKR", skip_crises=False)
    rp.HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, periods = [1, 3, 13, 26, 39, 52, 65], moment = 2, regressors = "HKR", skip_crises=False)
    rp.HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, periods = [1, 3, 13, 26, 39, 52, 65], moment = 2, regressors = "SMB + HML + HKR", skip_crises=True)
    rp.HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, periods = [1, 3, 13, 26, 39, 52, 65], moment = 2, regressors = "HKR", skip_crises=True)
    print("Finished using pseudo-monthly returns!")


@announce_execution
def fig_h1b_vs_smb_kkhml(eret_we_agg, figfolder):
# Example function to get U.S. recessions data
    def get_us_recessions():
        return [
            (datetime(2007, 12, 1), datetime(2009, 6, 30)),
            (datetime(2020, 2, 1), datetime(2020, 4, 30))
        ]
    
    eret_we_agg = eret_we_agg.set_index('yw')
    plt.figure()
    factors = (eret_we_agg.loc[:, ['SMB', 'HKR']])
    factors = factors/100 + 1
    factors = factors.cumprod()
    h1b_date = pd.to_datetime("20200301", errors="coerce", format="%Y%m%d")
    year_week = h1b_date.strftime('%Y%U')
    factors = factors.mul(100 / factors.loc[int(year_week)])
    factors.index = factors.index // 100 + (factors.index % 100) / 53

    # Plot factors
    factors.loc[2006:2021].plot()
    plt.axvline(2020 + 1/6, color='r', linestyle='--')

    # Include recessions shading
    recessions = get_us_recessions()
    for start, end in recessions:
        plt.axvspan(start.year + start.timetuple().tm_yday / 365,
                    end.year + end.timetuple().tm_yday / 365,
                    color='gray', alpha=0.5)

    plt.xlabel("Year")
    plt.ylabel("Cumulative return (Normalized to 100 at March 1st 2020)")
    plt.title("Cumulative return of SMB and HKR factors")
    plt.savefig(figfolder + "h1b_vs_smb_kkhml.jpg", dpi=600)
    plt.close()

@announce_execution
def tex_fmb_results_statistics(eret_we_agg, figfolder):
    # Pick columns only in set {'MktRF', 'SMB', 'HML', 'HKR', 'RMW', 'CMA'}:
    desired_columns = {'Mkt.RF', 'SMB', 'HML', 'HKR', 'RMW', 'CMA', 'HKR_NSB', 'HKR_SB'}

    eret_we_agg = eret_we_agg.loc[:, [col for col in eret_we_agg.columns if col in desired_columns]]
        # Convert values from weekly to monthly:
    summary = eret_we_agg.describe()
    # Transpose the dataframe:
    summary = summary.T
    # Multiply the values by 4.35 of all the columns except count:
    summary.loc[:, summary.columns != 'count'] = summary.loc[:, summary.columns != 'count'] * 4.35
    # Represent count as integer
    summary['count'] = summary['count'].astype(int)

    summary['Sharpe'] = summary['mean'] / summary['std']

    summary[['mean', 'std', 'min', 'max', '25%', '50%', '75%']] =\
        summary[['mean', 'std', 'min', 'max', '25%', '50%', '75%']].applymap(to_percentage)

    summary = summary[['count', 'mean', 'std', 'Sharpe', 'min', '25%', '50%', '75%', 'max']]
    summary['Sharpe'] = summary['Sharpe'].round(3)
    # Show only 3 digits after the decimal point for the Sharpe ratio:
    summary['Sharpe'] = summary['Sharpe'].apply(lambda x: f"{x:.3f}")
    
    # Rename columns:
    summary = summary.rename(columns = {"count": "Count", "mean": "Mean", "std": "SD", "min": "Min", "25%": "25\\%", "50%": "50\\%", "75%": "75\\%", "max": "Max"})
    # Remove min and max columns
    summary = summary.drop(columns = ["Min", "Max"])
    # Show row names as a column:
    summary = summary.reset_index()
    summary = summary.rename(columns = {"index": "Factor"})
    
    table = summary
    filename = "summary_statistics"
    with open(figfolder + filename + ".html", 'w') as html_file:
            html_file.write(table.to_html())
    
    # Remove row where Factor is in set {'HKR_NSB', 'HKR_SB'}:
    table = table[~table['Factor'].isin({'HKR_NSB', 'HKR_SB'})]

    with open(figfolder + filename + ".tex", 'w') as tex_file:
            contents = table.to_latex(index = False)
            tex_file.write(contents)

def to_percentage(x):
    return f"{x * 100:.2f}\\%"

@announce_execution
def explore_fmb(fmb_list, figfolder):
    tex_fmb_results(fmb_list, figfolder)

def explore_stoxda(stoxda, cequity_mapper, topic_map, figfolder):
    # Plot the Amazon stock prices
    # amazon_graph(stoxda, figfolder)
    # Format date and create 'y' and 'ym' columns

    stox, _ = preprocess_stoxda(stoxda, cequity_mapper, topic_map)
    plot_moment(stox, figfolder, "kurtosis", "Y", asset_weighted = False)
    plot_moment(stox, figfolder, "kurtosis", "Y", asset_weighted = True)
    plot_moment(stox, figfolder, "skewness", "Y", asset_weighted = False)
    #plot_moment(stoxda, cequity_mapper, topic_map, figfolder, "kurtosis", "M")
    #plot_moment(stoxda, cequity_mapper, topic_map, figfolder, "skewness", "M")
    return None

@announce_execution
def preprocess_stoxda(stoxda, cequity_mapper, topic_map):
        stoxda['date'] = pd.to_datetime(stoxda['date'])
        stoxda['y'] = stoxda['date'].dt.year
        stoxda['ym'] = stoxda['y']*100 + stoxda['date'].dt.month
        topic_map = topic_map.assign(me=lambda x: x['csho'] * x['prcc_f']) #groupby(['PERMNO', hue_var])
        stox = pd.merge(stoxda, cequity_mapper, left_on=['PERMNO', 'y'], right_on=['PERMNO', 'year'], how='left')
        stox = stox[stox['crit_ALL'] == 1]
        stox = pd.merge(stox, topic_map, left_on=['PERMNO', 'y'], right_on=['LPERMNO', 'year'], how='left')
        stox.set_index('date', inplace=True)
        return stox, topic_map

from stargazer_c.stargazer import Stargazer

def tex_fmb_results(fmb_list, figfolder):

    stargazer = Stargazer(fmb_list)
    stargazer.significant_digits(5)
    stargazer.title("Fama-MacBeth Regressions of Portfolio Weekly Excess Returns")
    stargazer.covariate_order(['HKR', 'MktRF', 'SMB', 'HML', 'RMW', 'CMA', 'Intercept'])
    stargazer.show_degrees_of_freedom(False)
    stargazer.show_f_statistic = False
    stargazer.show_residual_std_err = False
    stargazer.table_label = "tab:fmb_results"
    # stargazer.show_t_statistic = True 
    # stargazer.show_footer = False
    stargazer.dep_var_name = "Dep. var: Portfolio weekly excess return - "
    # Create a vector with "model_1", "model_2", with the length of fmb_list:
    model_vector = ["Mkt", "FF3", "FF5"]
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
    # Find the 1st percentiles of KKR:
    kkr_1 = topic_map['KKR'].quantile(0.01)
    # Find the 99th percentile of KKR:
    kkr_99 = topic_map['KKR'].quantile(0.99)

    for i, value in enumerate(unique_values):
        # Filter data for each category of 'ind12'
        data_filtered = topic_map[topic_map['ind12'] == value]
        
        # Plot histogram on the respective subplot
        sns.histplot(data=data_filtered, x='KKR', bins=15, kde=False, ax=axes[i], stat='probability')
        axes[i].set_title(f'{value}', fontsize=18)
        axes[i].set_xlabel('KKR')
        axes[i].set_ylabel('Probability')
        axes[i].set_xlim(kkr_1, kkr_99)  # Set x-axis limits for consistency

    
    plt.tight_layout()
    # Save the figure before displaying it
    plt.savefig(figfolder + "hist_kk_by_ind12.png", dpi=600)
    plt.close()

@announce_execution
def fig_histogram_kk(topic_map, figfolder):
    plt.figure()
    sns.histplot(data=topic_map, x='KKR', bins=20, kde=False, stat='probability')  # Adjust bins as needed
    plt.title('Histogram of KKR')
    plt.xlabel('Values of KKR')
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
            labs(x="N-tiles of Intangible Capital Risk measured by KKR",
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
    # Define kkpt_intensity as the ratio of K_int_Know to at:
    topic_map_positivekkpt['kkpt_intensity'] = topic_map_positivekkpt['K_int_Know'] / topic_map_positivekkpt['at'] 
    # Define kkpt_ntile as the quantile of kkpt_intensity:
    topic_map_positivekkpt['kkpt_ntile'] = topic_map_positivekkpt.groupby('year')["kkpt_intensity"].transform(lambda x: pd.qcut(x, nt, labels=False, duplicates='drop'))
    
    firms_by_kk = (topic_map_positivekkpt
                .groupby(['ntile_kk', 'kkpt_ntile'])
                .size()
                .reset_index(name='count'))

    plot = (ggplot(firms_by_kk, aes(x='ntile_kk', y='kkpt_ntile', fill='count')) +
            geom_tile(aes(fill='count')) +  # Use geom_tile for heatmap squares
            geom_text(aes(label='round(count, 2)'), size=20) +  # Add text labels
            scale_fill_gradient2(low="white", high="red", mid="pink", midpoint=firms_by_kk['count'].mean()) +  # Gradient fill
            labs(x="N-tiles of Knowledge Capital Risk measured by KKR",
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
                filter([col for col in topic_map.columns if col.startswith("topic")]+["xir_cumsum", "KKR"]))
    patent_correlations = patentcor.corr(method = "spearman").loc["xir_cumsum", [col for col in patentcor.columns if (col.startswith("topic") or col == "KKR")]]
    return patent_correlations

@announce_execution
def df_skill_correlations(topic_map):
    skillcor = (topic_map.
                filter([col for col in topic_map.columns if col.startswith("topic")]+["Skill"]))
    # Create an array with the correlation of each column that starts with "topic" with column "Skill":
    skill_correlations = skillcor.corr(method = "spearman").loc["Skill", [col for col in skillcor.columns if (col.startswith("topic") or col == "KKR")]]
    return skill_correlations

@announce_execution
def fig_share_dominant_kk_by_ind(topic_map, figfolder):
    maxkk = max(topic_map["ntile_kk"])
    ntile_name = get_quantile_term(maxkk)

    firms_by_ind = (topic_map
                    .loc[topic_map["ntile_kk"]==maxkk]
                    .groupby(["year", "ind12"])
                    .agg(count = ("year", "size"),
                        totalat = ("at", "sum"))
                    .reset_index(inplace = False))
    # Convert year and ind12 from index to columns:
    
    stackedplot_n = sns.barplot(data=firms_by_ind, x='year', y='count', hue='ind12', dodge=False)
    stackedplot_n.set_ylabel(f'Firm count in upper {ntile_name}')
    stackedplot_n.set_xlabel('Year')
    stackedplot_n.set_title(f'Firm count by Industry in upper {ntile_name}')
    stackedplot_n.legend(title='Industry', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)

    # Save plot above to "stackedplot_at.png" inside the "figfolder" directory:
    stackedplot_n.figure.savefig(figfolder + "stackedplot_n.png", bbox_inches='tight', dpi=300)
    
    plt.close()

    # Do the same for totalat:
    stackedplot_at = sns.barplot(data=firms_by_ind, x='year', y='totalat', hue='ind12', dodge=False)
    stackedplot_at.set_ylabel(f'Total assets of all firms in upper {ntile_name}')
    stackedplot_at.set_xlabel('Year')
    stackedplot_at.set_title(f'Total assets of all firms by Industry in upper {ntile_name}')
    stackedplot_at.legend(title='Industry', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)

    # Save plot above to "stackedplot_at.png" inside the "figfolder" directory:
    stackedplot_at.figure.savefig(figfolder + "stackedplot_at.png", bbox_inches='tight', dpi=300)

    plt.close()

@announce_execution
def tex_sample_topic_loadings(topic_map, figfolder):
    np.random.seed(139)
    sample_topics = (topic_map.
                    filter([col for col in topic_map.columns if col.startswith("topic")]+["conm", "KKR", "year"]).
                    sample(n = 10).
                    sort_values(by = "conm").
                    rename(columns = {"conm": "Company_Name"}).
                    apply(lambda x: round(x, 3) if (x.name.startswith('topic') or x.name == 'KKR') else x, axis = 0))

    personal_stargazer(sample_topics, figfolder, "sample_topics.tex", "tab:sample_topics", "Sample of Topic Loadings")

@announce_execution
def tex_average_topic_loadings_by_high_tech(topic_map, figfolder, nt):
    bytech = topic_map.dropna(subset = "hi_tech")
            
    bytech = (bytech.
            filter([col for col in bytech.columns if col.startswith("topic")]+["hi_tech", "KKR"]).
            groupby("hi_tech").
            agg(lambda x: round(x.mean(), 3) if (x.name.startswith('topic') or x.name == 'KKR') else x))

    personal_stargazer(bytech, figfolder, "bytech.tex", "tab:bytech", "Average Topic Loadings by High Tech")
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
    df_filtered = df.filter(regex='^year|topic_.*|KKR')
    
    # Calculate the average topic intensity for each year
    avg_df = df_filtered.groupby('year').mean().reset_index()
    
    # Reshape the DataFrame from wide to long format
    long_df = pd.melt(avg_df, id_vars=['year'], var_name='topic', value_name='intensity')
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=long_df, x='year', y='intensity', hue='topic')
    # Set ylim
    plt.ylim(0.1, 0.5)
    
    # Setting labels and title
    plt.xlabel("Year")
    plt.ylabel("Topic Intensity")
    plt.title("Mean Knowledge Capital Risk by Year")
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

@announce_execution
def calc_svar(stox, frequency = 'W'):
    # Let statfunc be the sum of squared returns:
    statfunc = lambda x: np.sum(x ** 2)
    moment_df = stox.groupby(['PERMNO']).resample(frequency)['RET'].apply(statfunc)
    moment_df = moment_df.reset_index()
    svar_qwe = rp.to_quadriweekly(moment_df, 'RET')
    return svar_qwe

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
    if asset_weighted:
        aw_suffix = "asset-weighted"
        aw_suffix_short = "aw"
        stoxme_df = stox.reset_index()[['PERMNO', 'date', 'me']]
        moment_df = moment_df.merge(stoxme_df, on=['PERMNO', 'date'], how='left')
        moment_df = moment_df.dropna(subset=['RET', 'me'])
        stox_by_kk = (moment_df.groupby([hue_var, 'date'])
                            .apply(lambda x: pd.Series({'avg_kurt': (x['RET'] * x['me']).sum() / x['me'].sum()}))
                            .reset_index())
    else:
        aw_suffix = "non-asset-weighted"
        aw_suffix_short = "naw"
        stox_by_kk = (moment_df.reset_index().groupby([hue_var, 'date'])
                        .agg(avg_kurt=('RET', 'mean'))
                        .reset_index())
        #moment_df= moment_df.reset_index()
       
    large_palette = sns.color_palette('husl', 8)
    # Find the two lowest and two highest values of ntile_kk:

    # Filter stox_by_kk to only include ntile_kk values equal to the two lowest and two highest values of ntile_kk
    if hue_var == 'ntile_kk':
        stox_by_kk = keep_extremes(stox_by_kk, hue_var, 1)
        if max(stox_by_kk[hue_var]) == 10:
            ntilename = "Decile"
        else:
            ntilename = hue_var
    if hue_var == 'max_topic':
        # Filter only rows where max_topic is between 0 and 7 (inclusive):
        stox_by_kk = stox_by_kk[stox_by_kk[hue_var].between(0, 7)]
    sns.lineplot(data=stox_by_kk, x='date', y='avg_kurt', hue=hue_var, palette=large_palette)
    plt.title(f"Annual {moment_name} of daily returns across firms ({aw_suffix})")
    plt.xlabel("Year-Month")
    plt.ylabel(f"{moment_name}")
    plt.legend(title=ntilename, fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{figfolder}/{moment_name}_by_{hue_var}_{frequency}_{aw_suffix_short}.jpg", dpi=600)
    plt.clf()
    return None

def keep_extremes(df, hue_var, num):
    df = df.copy()
    ntile_kk_values = df[hue_var].unique()
    sorted_values = sorted(ntile_kk_values)
    lowest_two = sorted_values[:num]
    highest_two = sorted_values[-num:]
    keep = lowest_two + highest_two
    df = df[df[hue_var].isin(keep)]
    return df

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
    
    # Create we_ret_bybin with a 3-month moving average of sderet:
    we_ret_bybin['sderet_4wa'] = we_ret_bybin.groupby('ntile_kk')['sderet'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

    # Same for 12-week moving average of sderet:
    we_ret_bybin['sderet_12wa'] = we_ret_bybin.groupby('ntile_kk')['sderet'].transform(lambda x: x.rolling(window=12, min_periods=1).mean())

    qt_ret_bygroup = (we_ret_bybin
                      .groupby('ntile_kk')
                      .apply(lambda x: x.assign(eret3ma=x['eret'].rolling(window=13, min_periods=1).mean()))
                      .reset_index(drop=True))
    
    # The following plot is commented out because it does not convey any useful information
    # plt.figure()
    # sns.lineplot(data=we_ret_bybin, x='yw', y='eret', hue='ntile_kk')
    # plt.xlabel("Year-month")
    # plt.ylabel("Asset-weighted weekly returns")
    # plt.legend(fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.savefig(figfolder + "aw_wr.jpg", dpi=600)
    # plt.clf()

    # The following plot is commented out because it does not convey any useful information    
    # plt.figure()
    # sns.lineplot(data=qt_ret_bygroup, x='yw', y='eret3ma', hue='ntile_kk')
    # plt.xlabel("Year-week")
    # plt.ylabel("Asset-weighted weekly returns, 3MA")
    # plt.legend(fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.savefig(figfolder + "aw_wr_3ma.jpg", dpi=600)
    # plt.clf()
    
    data_to_show = keep_extremes(we_ret_bybin, 'ntile_kk', 1)
    plt.figure()
    sns.lineplot(data=data_to_show, x='yw', y='eret_accum', hue='ntile_kk')
    plt.xlabel("Year-week")
    plt.ylabel("Asset-weighted accumulated weekly returns")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(figfolder + "aw_accum_wr.jpg", dpi=600)
    plt.clf()

    # The following plot is commented out because it does not convey any useful information
    # plt.figure()
    # sns.lineplot(data=we_ret_bybin, x='yw', y='sderet', hue='ntile_kk')
    # plt.xlabel("Year-week")
    # plt.ylabel("Weekly standard deviation of returns by n-tile")
    # plt.legend(fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.ylim(0.02,0.14)
    # plt.savefig(figfolder + "sd_eret.jpg", dpi=600)
    # plt.clf()
    
    plt.figure()
    sns.lineplot(data=data_to_show, x='yw', y='sderet_4wa', hue='ntile_kk')
    plt.xlabel("Year-week")
    plt.ylabel("Weekly standard deviation of returns by n-tile (4-week MA)")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0.02,0.14)
    plt.savefig(figfolder + "sd_eret_4ma.jpg", dpi=600)
    plt.clf()

    plt.figure()
    sns.lineplot(data=data_to_show, x='yw', y='sderet_12wa', hue='ntile_kk')
    plt.xlabel("Year-week")
    plt.ylabel("Weekly standard deviation of returns by n-tile (12-week MA)")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0.02,0.14)
    plt.savefig(figfolder + "sd_eret_12ma.jpg", dpi=600)
    plt.clf()

    return None    

@announce_execution
def explore_betas(df_betas, quantiles, figfolder):
    # Set index names
    pf_type = df_betas.head().index.names[0]
    df_betas.index.names = ['pfname', 'yw']

    # Calculate mean and standard deviation for each pfname
    mean_df = df_betas.groupby('pfname').mean()
    def compute_t_stat(group):
        t_stats = group.apply(lambda x: ttest_1samp(x, 0).statistic)
        return t_stats

    # Calculate t-statistics for each pfname
    t_stat_df = df_betas.groupby('pfname').apply(compute_t_stat)

    # Merge mean and t-statistics dataframes
    betas_exh = mean_df.join(t_stat_df, lsuffix='_mean', rsuffix='_tstat')
    
    # Keep only the columns related to HKR and eretw
    columns_to_keep = [col for col in betas_exh.columns if 'HKR' in col or 'eretw' in col or "alpha" in col]
    filtered_df = betas_exh[columns_to_keep]
    filtered_df.index.names = ['pfname']
    
    if quantiles == 10:
        to_keep = [0, 5, 9]
    elif quantiles == 5:
        to_keep = [0, 2, 4]
    elif quantiles == 4:
        to_keep = [0, 1, 2, 3]
    elif quantiles == 3:
        to_keep = [0, 1, 2]
    else:
        raise ValueError("Invalid quantiles. Choose 10, 5, 4, or 3.")
    to_keep = list(map(str, to_keep))
        
    # Convert the index to a DataFrame to manipulate it
    index_df = filtered_df.index.to_frame(index=False)
    
    # Add leading zeros to pfname
    index_df['pfname'] = index_df['pfname'].astype(str).str.zfill(3)
    
    # Apply the function to create long_pfname
    index_df['long_pfname'] = index_df.apply(lambda row: create_long_pfname(row['pfname'], pf_type), axis=1)    
    # Filter index_df based on specific conditions
    filtered_index_df = index_df[index_df['pfname'].str[0].isin(to_keep)]
    
    # Reset index for merging
    filtered_index_df = filtered_index_df.set_index('pfname')
    filtered_index_df = filtered_index_df.reset_index()
    filtered_df = filtered_df.reset_index()
    
    # Strip leading zeros for merging
    filtered_df['pfname'] = filtered_df['pfname'].astype(str).str.lstrip('0')
    filtered_index_df['pfname'] = filtered_index_df['pfname'].astype(str).str.lstrip('0')
    
    # Merge the dataframes
    betas_exh = pd.merge(filtered_df, filtered_index_df, on='pfname')
    
    # Set the new index and drop the redundant column
    betas_exh = betas_exh.drop(columns=['pfname'])
    betas_exh.rename(columns={'long_pfname': 'PF', "HKR_mean": "beta_HKR", "eretw_mean": "RET", "eretw_tstat": "t(RET)", "alpha_mean": "alpha", "alpha_tstat": "t(alpha)", "HKR_tstat": "t(beta_HKR)"}, inplace=True)
    # Get only columns ['PF', 'alpha', 't(alpha)', 'beta_HKR', 't(beta_HKR)', 'RET', 't(RET)']:
    betas_exh = betas_exh[['PF', 'alpha', 't(alpha)', 'beta_HKR', 't(beta_HKR)', 'RET']]
    # # Convert alpha and RET to percentage, round to 2 decimal places, and add percentage sign:
    betas_exh['alpha'] = (betas_exh['alpha'] * 100).round(2).astype(str) + r'\%'
    betas_exh['RET'] = (betas_exh['RET'] * 100).round(2).astype(str) + r'\%'
    # Round the other columns to 3 decimal places. List them nominally:
    betas_exh['t(alpha)'] = betas_exh['t(alpha)'].round(3).apply(lambda x: f"{x:.3f}")
    betas_exh['t(beta_HKR)'] = betas_exh['t(beta_HKR)'].round(3).apply(lambda x: f"{x:.3f}")
    #betas_exh['t(RET)'] = betas_exh['t(RET)'].round(3).apply(lambda x: f"{x:.3f}")
    betas_exh['beta_HKR'] = betas_exh['beta_HKR'].round(3).apply(lambda x: f"{x:.3f}")


    betas_exh.columns = ['PF',r'$\alpha$',r't($\alpha$)',r'$\beta_\text{HKR}$',r't($\beta_\text{HKR}$)','RET']
    filename = "betas"
    with open(figfolder + filename + ".html", 'w') as html_file:
            html_file.write(betas_exh.to_html())

    with open(figfolder + filename + ".tex", 'w') as tex_file:
            contents = betas_exh.to_latex(escape = False, index = False)
            tex_file.write(contents)
    
    return None

def create_long_pfname(pfname, pf_type):
    if len(pfname) != 3:
        return pfname
    elif pf_type == 'pfkki3me3mb':
        first_digit, second_digit, third_digit = pfname[0], pfname[1], pfname[2]
        second_char = 'S' if second_digit == '1' else ('M' if second_digit in '2' else 'B')
        third_char = 'L' if third_digit == '1' else ('M' if third_digit == '2' else ('H' if third_digit == '3' else ''))
        return f"{second_char}/{third_char}/{first_digit}"
    else:
        return pfname
    
def setup(quantiles, modelname, pfname, suffix = "_HKR_SB"):
    base_path = "/Users/pedrovallocci/Documents/PhD (local)/Research/Github/KnowledgeKRisk_10Ks/text/"
    dir_path = os.path.join(base_path, f"{modelname}_{quantiles}tiles_{pfname}{suffix}")
    os.makedirs(dir_path, exist_ok=True)
    figfolder = os.path.join(dir_path, "")
    add_innerkk_pf = not modelname.startswith("dicfull")
    cuts = np.linspace(0, 1, quantiles+1).tolist()
    print(f"Running model {modelname} with {quantiles} quantiles and {pfname} portfolio")
    return figfolder, add_innerkk_pf, cuts