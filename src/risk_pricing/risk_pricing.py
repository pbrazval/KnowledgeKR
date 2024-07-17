import os
import warnings
import time
import random
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.rolling import RollingOLS
from statsmodels.iolib.summary2 import summary_col
from linearmodels.panel.model import FamaMacBeth
from linearmodels.panel import generate_panel_data
from linearmodels.asset_pricing import LinearFactorModel, LinearFactorModelGMM
import visualization as viz
import data_loading as dl
import risk_pricing as rp
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf


def get_fiscal_year_we(yw):
    # Placeholder for the actual fiscal year calculation logic
    # Assuming yw is a year-week format integer
    year = yw // 100
    week = yw % 100
    # Adjust based on fiscal year logic
    fy = [y + 1 if w > 26 else y for y, w in zip(year, week)]
    return fy

def process_stoxwe(stoxwe_post2005short, cequity_mapper, topic_map, ffm, pfn, add_innerkk_pf, kki_cuts, log_returns):
    # Apply the conditions and choices to create the mb_group column
    if "CMA" in ffm.columns:
        case = "ff5"
    else:
        case = "ff3"
        
    stoxwe = (stoxwe_post2005short.
            assign(y=lambda x: x['yw'] // 100).
            merge(cequity_mapper, left_on=["PERMNO", "y"], right_on=["PERMNO", "year"], how="inner"))
    stoxwe = stoxwe[stoxwe['crit_ALL'] == 1]
    min_year = topic_map['year'].min()

    stoxwe = (stoxwe
            .merge(topic_map, left_on=["PERMNO", "y"], right_on=["LPERMNO", "year"], how="inner")
            .query('y >= @min_year')
            .merge(ffm, on = "yw", how = "left")
            .assign(eretw = lambda x: x['retw'] - x['RF'])
            .dropna(subset = ['retw'])
            .assign(fiscalyear = lambda x: rp.get_fiscal_year_we(x['yw']))
            .assign(mb = lambda x: (x['csho'] * x['prcc_f']) / x['ceq'], 
                    me = lambda x: x['csho'] * x['prcc_f'],
                    kk_share = lambda x: x['K_int_Know'] / x['ppegt'],
                    CUSIP8 = lambda x: x['cusip'].str[:-1]))
    pfs = (stoxwe
        .query('yw % 100 == 26')  # Filter rows where yw % 100 == 26
        .drop(columns=['cusip'])  # Drop the 'cusip' column
        .dropna(subset=['me', 'mb'])  # Drop rows where x'me' or 'mb' is NA
        .groupby('y')  # Group by 'y'
        .apply(lambda df: df.assign(  # Use apply to perform the following operations within each group
            med_NYSE_me=df.loc[df['exchg'] == 11, 'me'].median(),  # Calculate median of 'me' for exchg == 11
            med_NYSE_mb70p=np.quantile(df.loc[df['exchg'] == 11, 'mb'].dropna(), 0.7),  # 70th percentile of 'mb' for exchg == 11
            med_NYSE_mb30p=np.quantile(df.loc[df['exchg'] == 11, 'mb'].dropna(), 0.3)  # 30th percentile of 'mb' for exchg == 11
        ))
        .reset_index(drop=True)  # Reset index after groupby operation
        .drop_duplicates()  # In case the apply operation replicated rows
        .assign(me_group=lambda df: np.where(df['me'] < df['med_NYSE_me'], 1, 2)))
    
    conditions = [
        pfs['mb'] <= pfs['med_NYSE_mb30p'],  # Condition for mb_group = 1
        (pfs['mb'] > pfs['med_NYSE_mb30p']) & (pfs['mb'] <= pfs['med_NYSE_mb70p']),  # Condition for mb_group = 2
        pfs['mb'] > pfs['med_NYSE_mb70p']  # Condition for mb_group = 3
    ]

    # Choices corresponding to each condition
    choices = [1, 2, 3]

    # Apply the conditions and choices to create the mb_group column
    pfs['mb_group'] = np.select(conditions, choices, default=np.nan)
    if add_innerkk_pf:
        pfs = (pfs #.drop(columns=['med_NYSE_me', 'med_NYSE_mb30p', 'med_NYSE_mb70p'])  # Drop the columns used to create 'me_group' and 'mb_group'
                .assign(pf2me3mb=lambda x: 10 * x['me_group'] + x['mb_group'])
                .groupby('y')
                .apply(lambda df: df.assign(me_3tile=1+pd.qcut(df['me'], 3, labels=False, duplicates='raise'),
                                            mb_3tile=1+pd.qcut(df['mb'], 3, labels=False, duplicates='raise'),
                                            me_5tile=1+pd.qcut(df['me'], 5, labels=False, duplicates='raise'),
                                            mb_5tile=1+pd.qcut(df['mb'], 5, labels=False, duplicates='raise')))
                .assign(pf5me5mb=lambda x: 10 * x['me_5tile'] + x['mb_5tile'])
                .assign(pfkk2me3mb = lambda x: 100 * x['ntile_kk'] + 10 * x['me_group'] + x['mb_group'])
                .assign(pfkk3me3mb = lambda x: 100 * x['ntile_kk'] + 10 * x['me_3tile'] + x['mb_3tile'])
                .reset_index(drop=True)
                .groupby(['y', 'me_3tile', 'mb_3tile'])
                .apply(lambda df: df.assign(kkr_ntile_inner=pd.qcut(df['KKR'], kki_cuts, labels=False, duplicates='raise')))
                .reset_index(drop=True)
                .assign(pfkki3me3mb=lambda x: 100 * x['kkr_ntile_inner'] + 10 * x['me_3tile'] + x['mb_3tile'])
                .reset_index(drop=True)
                .loc[:, ['gvkey_x', 'pfkk3me3mb', 'pf2me3mb', 'pf5me5mb', 'pfkk2me3mb', 'pfkki3me3mb', 'fiscalyear', 'me_group']]
                .rename(columns={'gvkey_x': 'gvkey'})
                .assign(fiscalyear=lambda x: x['fiscalyear'] + 1)
        )
    else:
        pfs = (pfs #.drop(columns=['med_NYSE_me', 'med_NYSE_mb30p', 'med_NYSE_mb70p'])  # Drop the columns used to create 'me_group' and 'mb_group'
                .assign(pf2me3mb=lambda x: 10 * x['me_group'] + x['mb_group'])
                .groupby('y')
                .apply(lambda df: df.assign(me_3tile=1+pd.qcut(df['me'], 3, labels=False, duplicates='raise'),
                                            mb_3tile=1+pd.qcut(df['mb'], 3, labels=False, duplicates='raise'),
                                            me_5tile=1+pd.qcut(df['me'], 5, labels=False, duplicates='raise'),
                                            mb_5tile=1+pd.qcut(df['mb'], 5, labels=False, duplicates='raise')))
                .assign(pf5me5mb=lambda x: 10 * x['me_5tile'] + x['mb_5tile'])
                .assign(pfkk2me3mb = lambda x: 100 * x['ntile_kk'] + 10 * x['me_group'] + x['mb_group'])
                .assign(pfkk3me3mb = lambda x: 100 * x['ntile_kk'] + 10 * x['me_3tile'] + x['mb_3tile'])
                .reset_index(drop=True)
                .loc[:, ['gvkey_x', 'pfkk3me3mb', 'pf2me3mb', 'pf5me5mb', 'pfkk2me3mb', 'fiscalyear', 'me_group']]
                .rename(columns={'gvkey_x': 'gvkey'})
                .assign(fiscalyear=lambda x: x['fiscalyear'] + 1)
        )

    stoxwe_add = (stoxwe.copy()
                .rename(columns={'gvkey_x': 'gvkey'})
                .merge(pfs, on=['gvkey', 'fiscalyear'], how='inner'))
    
    if case == "ff3":
        pf_ret = (stoxwe_add.dropna(subset=['eretw', 'me'])
            .groupby(['yw', pfn])  # Replace pfn with the actual column name
            .apply(lambda df: pd.Series({
                'eret': (df['eretw'] * df['me']).sum() / df['me'].sum(),
                'Mkt.RF': df['Mkt-RF'].mean(),
                'SMB': df['SMB'].mean(),
                'HML': df['HML'].mean(),
                'RF': df['RF'].mean()
            }))
            .reset_index())
    else:
        pf_ret = (stoxwe_add.dropna(subset=['eretw', 'me'])
            .groupby(['yw', pfn])  # Replace pfn with the actual column name
            .apply(lambda df: pd.Series({
                'eret': (df['eretw'] * df['me']).sum() / df['me'].sum(),
                'Mkt.RF': df['Mkt-RF'].mean(),
                'SMB': df['SMB'].mean(),
                'HML': df['HML'].mean(),
                'CMA': df['CMA'].mean(),
                'RMW': df['RMW'].mean(),
                'RF': df['RF'].mean()
            }))
            .reset_index())
    
    # # Calculate HKR returns
    
    max_kknt = max(stoxwe_add['ntile_kk'])
    min_kknt = min(stoxwe_add['ntile_kk'])
    
    HKR_NSB_ret = (stoxwe_add.dropna(subset=['KKR'])
                .groupby(['yw', 'ntile_kk'])
                .apply(lambda df: pd.Series({
                    'eret': (df['eretw'] * df['me']).sum() / df['me'].sum()}))
                .reset_index()
                .pivot_table(index='yw', columns='ntile_kk', values='eret', aggfunc='mean')
                .rename(columns=lambda x: f'kk{x}')
                .assign(HKR_NSB=lambda df: df[f'kk{max_kknt}'] - df[f'kk{min_kknt}'])
                [['HKR_NSB']]
                .reset_index())
    
    HKR_ret = (stoxwe_add
            .loc[:, ['yw', 'ntile_kk', 'me_group', 'KKR', 'eretw', 'me']]
            .dropna(subset=['KKR'])
            .groupby(['yw', 'ntile_kk', 'me_group'])
            .apply(lambda df: pd.Series({
                'eret': (df['eretw'] * df['me']).sum() / df['me'].sum()}))
            .reset_index()
            .groupby(['yw', 'ntile_kk'])
            .agg(eret_mean=('eret', 'mean'))
            .reset_index()
            .pivot_table(index='yw', columns='ntile_kk', values='eret_mean', aggfunc='mean')
            .rename(columns=lambda x: f'kk{x}')
            .assign(HKR=lambda df: df[f'kk{max_kknt}'] - df[f'kk{min_kknt}'])
            .reset_index()
            .loc[:, ['yw', 'HKR']])
    
    HKR_ret.columns.name = None
    
    eret_we = (pf_ret.merge(HKR_ret, on='yw', how='inner')
               .merge(HKR_NSB_ret, on='yw', how='inner')
            .rename(columns={'eret': 'eretw'})
            .dropna()
            .reset_index(drop=True))
    
    eret_we_pct = convertLogReturnsToPercentage(eret_we, log_returns)
    
    return eret_we, stoxwe_add, eret_we_pct

def convertLogReturnsToPercentage(eret_we, log_returns):
    eret_we_copy = eret_we.copy()
    if not log_returns:
        print("Now converting log returns to percentage")
        exclude_cols = {'index', 'date', 'yw'}
        exclude_cols.update([col for col in eret_we_copy.columns if col.startswith('pf')])
        for col in eret_we_copy.columns:
            if col not in exclude_cols:
                print("New version of log_return_to_percentage")
                eret_we_copy[col] =  (np.exp(eret_we[col]) - 1) * 100 # eret_we_copy[col].apply(log_return_to_percentage)
    return eret_we_copy

def log_return_to_percentage(log_return):
    result = (np.exp(log_return) - 1) * 100
    print("New version of log_return_to_percentage")
    return result

def famaMacBeth(eret_we, pfname, formula = None, window_size = 52):
    
    if formula is None:
        if "CMA" in eret_we.columns:
            case = "ff5"
            formula = "eretw ~ 1 + MktRF + SMB + HML + CMA + RMW + HKR"
        else:
            case = "ff3"
            formula = "eretw ~ 1 + MktRF + SMB + HML + HKR"
    elif formula == "eretw ~ 1 + MktRF + HKR":
        case = "ff1"
    else:
        raise ValueError("Invalid formula. Please provide a valid formula for the regression.")
    
    eret_we2 = add_constant(eret_we, prepend=False)
    eret_we2.set_index('yw', inplace=True)
    eret_we2.rename(columns={'const': 'alpha', 'Mkt.RF': 'MktRF'}, inplace=True)
    results_list = []

    # Iterate over each unique value of pfkk3me3mb
    for pf_name in eret_we2[pfname].unique():
        # Subset the DataFrame for the current pfkk3me3mb
        subset_data = eret_we2[eret_we2[pfname] == pf_name]
        
        # Initialize and fit the RollingOLS model
        mod = RollingOLS.from_formula(formula, data=subset_data, window=window_size, expanding=False)
        rres = mod.fit(cov_type="HC0", method="pinv", params_only=True)
        
        # Assuming your DataFrame's index contains the time window identifier 'yw'
        # Extract coefficients for each window and create a DataFrame
        for idx, params in rres.params.iterrows():
            if case == "ff3":
                results_list.append([idx, pf_name, params['Intercept'], params['MktRF'], params['SMB'], params['HML'], params['HKR']])
            elif case == "ff5":
                results_list.append([idx, pf_name, params['Intercept'], params['MktRF'], params['SMB'], params['HML'], params['HKR'], params['CMA'], params['RMW']])
            elif case == "ff1":
                results_list.append([idx, pf_name, params['Intercept'], params['MktRF'], params['HKR']])

    # Convert the list of results into a DataFrame
    if case == "ff3":
        results_df = pd.DataFrame(results_list, columns=['yw', pfname, 'alpha', 'MktRF', 'SMB', 'HML', 'HKR'])
    elif case == "ff5":
        results_df = pd.DataFrame(results_list, columns=['yw', pfname, 'alpha', 'MktRF', 'SMB', 'HML', 'HKR', 'CMA', 'RMW'])
    elif case == "ff1":
        results_df = pd.DataFrame(results_list, columns=['yw', pfname, 'alpha', 'MktRF', 'HKR'])
        
    results_df = results_df.merge(eret_we[['yw', pfname, 'eretw']], on=['yw', pfname], how='left')
    # Added today: Jul 1st 2024
    results_df.dropna(inplace=True) 
    df_betas = results_df.set_index([pfname, 'yw'], inplace=False)
    print("New kernel")
    # Multiply all the columns (if present) by 100 to get the percentage values, if they belong to the list: 'alpha', 'MktRF', 'SMB', 'HML', 'HKR', 'CMA', 'RMW'
    fmb = FamaMacBeth.from_formula(formula, df_betas).fit(cov_type='kernel')
    return fmb, df_betas

def famaMacBethFull(stoxwe_post2005short, cequity_mapper, topic_map, ffm, pfname, log_returns = False, formula = None, kki_cuts = [0, 0.2, 0.4, 0.6, 0.8, 1], window_size = 52, add_innerkk_pf = False):
    eret_we, stoxwe_add, eret_we_pct = rp.process_stoxwe(stoxwe_post2005short, cequity_mapper, topic_map, ffm, pfname, add_innerkk_pf, kki_cuts, log_returns = log_returns)
    print("Finished processing stoxwe")
    fmb, df_betas = rp.famaMacBeth(eret_we, pfname, formula = formula, window_size=window_size)
    print("Finished Fama-MacBeth")
    return fmb, df_betas, eret_we, stoxwe_add, eret_we_pct

def famaMacBethMultiCase(pfname, add_innerkk_pf, cuts, cequity_mapper, ff3fw, ff5fw, stoxwe_orig, topic_map, log_returns = False):
    print("Running Fama-MacBeth regressions: 5 factors")
    fmb_5, df_betas5, eret_we5, stoxwe_add, eret_we_pct5 = rp.famaMacBethFull(stoxwe_orig, cequity_mapper, topic_map, ff5fw, pfname, kki_cuts = cuts, window_size = 52*2, add_innerkk_pf = add_innerkk_pf, log_returns = log_returns)
    print("Running Fama-MacBeth regressions: 3 factors")
    fmb_3, df_betas3, eret_we3, _, eret_we_pct3 = rp.famaMacBethFull(stoxwe_orig, cequity_mapper, topic_map, ff3fw, pfname, kki_cuts = cuts, window_size = 52*2, add_innerkk_pf = add_innerkk_pf, log_returns = log_returns)
    fmb_1, df_betas1 = rp.famaMacBeth(eret_we5, pfname, formula = "eretw ~ 1 + MktRF + HKR", window_size=52*2)
    print("Running Fama-MacBeth regressions: 1 factor")
    return fmb_5, fmb_3, fmb_1, df_betas5, df_betas3, df_betas1, eret_we5, eret_we3, stoxwe_add, eret_we_pct5, eret_we_pct3

def pseudo_monthly(eret_we_agg):
    # Keep only columns whose names match one of the set: date, eretw, Mkt.RF, SMB, HML, RMW, CMA, RF, HKR, HKR_NSB
    columns_to_keep = {'date', 'eretw', 'Mkt.RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'HKR', 'HKR_NSB', 'SVAR'}
    # Drop columns index, yw, and any other column whose name is not in columns_to_keep
    eret_we_agg = eret_we_agg[eret_we_agg.columns.intersection(columns_to_keep)]

    columns_to_aggregate = ['eretw', 'Mkt.RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'HKR', 'HKR_NSB', 'SVAR']

    # Apply the aggregation function
    pseudo_monthly = rp.to_quadriweekly(eret_we_agg, columns_to_aggregate)
    return pseudo_monthly

def to_quadriweekly(df, columns_to_aggregate):
    # Create a grouping variable
    df['group'] = df.index // 4
    
    # Define aggregation rules
    agg_rules = {'date': ('date', 'last')}
    
    for col in columns_to_aggregate:
        if col in df.columns:
            agg_rules[col] = (col, 'sum')
    
    # Aggregate the groups
    pseudo_monthly = df.groupby('group').agg(**agg_rules).reset_index(drop=True)
    
    return pseudo_monthly

def ret_nperiods_ahead(eret_we_agg, n, moment = 1, var = 'Mkt.RF'):
    
    if moment == 1:
        var = "Mkt.RF"
        var_clean = "MktRF"
    elif moment == 2:
        var = "SVAR"
        var_clean = "SVAR"
        var_long = "SVAR"
    else:
        raise ValueError("Invalid moment value. Please provide a valid moment (1 or 2).")

    eret_we_agg[f'{var_clean}_{n}w'] = (eret_we_agg[f'{var}']
            .rolling(window=n)
            .sum()
            .shift(-n))
    return eret_we_agg

def HKR_vs_mktrf_qwe(eret_qwe_agg, figfolder, moment = 1, periods = [1, 3, 13, 26, 39, 52, 65], regressors = "SMB + HML + HKR", skip_crises=False):
    eret_qwe = eret_qwe_agg.copy()
    
    if moment == 1:
        var = "Mkt.RF"
        var_clean = "MktRF"
        var_long = "Excess Market Return"
    elif moment == 2:
        var = "SVAR"
        var_clean = "SVAR"
        var_long = "SVAR"
    else:
        raise ValueError("Invalid moment value. Please provide a valid moment (1 or 2).")
    
    if skip_crises:
        eret_qwe = eret_qwe[eret_qwe['date'].dt.year >= 2009]
        suffix = "skc"
    else:
        suffix = "noskc"
    
    for period in periods:
        eret_qwe = ret_nperiods_ahead(eret_qwe, period, moment = moment)
    
    rename_dict = {f'{var_clean}_{period}w': f'{var_clean}_{period * 4}w' for period in periods}
    eret_qwe.rename(columns=rename_dict, inplace=True)

    models = []
    model_names = []

    for period in periods:
        period_renamed = period * 4
        formula = f"{var_clean}_{period_renamed}w ~ {regressors}"
        model = smf.ols(formula=formula, data=eret_qwe).fit(cov_type= 'HAC', cov_kwds={'maxlags': period})
        models.append(model)
        model_names.append(f'{period_renamed}wa')
    
    reg_order = [var.strip() for var in regressors.split('+')]

    summary = summary_col(
        models,
        model_names=[rf"$q={period}$" for period in periods],
        stars=True,
        float_format='%0.4f',
        regressor_order=reg_order,
        drop_omitted=True
    )

    tex_content = summary.as_latex(label = "tab:HKR_vs_mktrf")
    tex_content = tex_content.replace(r"\$", "$")
    # Substitute substring "label{}" by "label{tab:HKR_vs_mktrf}":
    tex_content = re.search(r'\\begin{tabular}.*?\\end{tabular}', tex_content, re.DOTALL).group()
    #tex_content = tex_content.replace("caption\{\}", "caption\{HKR vs. {var} Weeks Ahead: Summary Statistics\}")
    # Remove the footnotes section from the LaTeX content
    tex_content = tex_content.split("\\bigskip")[0]
    
    # Define the file path for the tex file
    tex_file_path = figfolder + f"HKR_vs_{var_clean}_{'_'.join(reg_order)}_{suffix}.tex"
    
    # Write the modified summary to the tex file
    with open(tex_file_path, 'w') as tex_file:
        tex_file.write(tex_content)

def HKR_vs_mktrf(eret_we_agg, skip_crises=True):
    if skip_crises:
        eret_we_agg = eret_we_agg[eret_we_agg['yw'] > 200900]
        eret_we_agg = eret_we_agg[eret_we_agg['yw'] < 202012]
    eret_we_agg = ret_nperiods_ahead(eret_we_agg, 4)
    eret_we_agg = ret_nperiods_ahead(eret_we_agg, 12)
    eret_we_agg = ret_nperiods_ahead(eret_we_agg, 52)
    eret_we_agg = ret_nperiods_ahead(eret_we_agg, 104)
    eret_we_agg = ret_nperiods_ahead(eret_we_agg, 156)
    eret_we_agg = ret_nperiods_ahead(eret_we_agg, 208)
    eret_we_agg = ret_nperiods_ahead(eret_we_agg, 260)

    eret_we_agg.rename(columns = {'Mkt.RF_4w': 'MktRF_4w', 
                                'Mkt.RF_12w': 'MktRF_12w', 
                                'Mkt.RF_52w': 'MktRF_52w', 
                                'Mkt.RF_104w': 'MktRF_104w', 
                                'Mkt.RF_156w': 'MktRF_156w',
                                'Mkt.RF_208w': 'MktRF_208w', 
                                'Mkt.RF_260w': 'MktRF_260w'
                                }, inplace=True)

    summary = (summary_col([smf.ols(formula="MktRF_4w ~ SMB + HML + HKR", data=eret_we_agg).fit(cov_type='HAC', cov_kwds={'maxlags': 4}),  # cov_type='HAC', cov_kwds={'maxlags': 4}: Should be added?
                            smf.ols(formula="MktRF_12w ~ SMB + HML + HKR", data=eret_we_agg).fit(cov_type='HAC', cov_kwds={'maxlags': 12}),#fit(cov_type='HAC', cov_kwds={'maxlags': 12}), 
                            smf.ols(formula="MktRF_52w ~ SMB + HML + HKR", data=eret_we_agg).fit(cov_type='HAC', cov_kwds={'maxlags': 52}),#fit(cov_type='HAC', cov_kwds={'maxlags': 52}), 
                            smf.ols(formula="MktRF_104w ~ SMB + HML + HKR", data=eret_we_agg).fit(cov_type='HAC', cov_kwds={'maxlags': 104}),#fit(cov_type='HAC', cov_kwds={'maxlags': 104}), 
                            smf.ols(formula="MktRF_156w ~ SMB + HML + HKR", data=eret_we_agg).fit(cov_type='HAC', cov_kwds={'maxlags': 156}),#fit(cov_type='HAC', cov_kwds={'maxlags': 156})smf.ols(formula="MktRF_208w ~ SMB + HML + HKR", data=eret_we_agg).fit(cov_type='HAC', cov_kwds={'maxlags': 208}), 
                            smf.ols(formula="MktRF_260w ~ SMB + HML + HKR", data=eret_we_agg).fit(cov_type='HAC', cov_kwds={'maxlags': 260})
                            ],  # List of regression result objects
                        model_names=['4wa', '12wa', '52wa', '104wa', '156wa', '208wa', '260wa'],  # Names for each model
                        stars=True,  # Include significance stars
                        float_format='%0.4f',  # Format for displaying coefficients
                        regressor_order=['HKR', 'SMB', 'HML'],  # Order of variables in the table
                        drop_omitted=False))  # Drop omitted variables
    return summary

def gmm(eret_we, factors = ['Mkt.RF', 'SMB', 'HML', 'HKR'], formula = "Mkt.RF + HKR", quadriweekly = True):
    # Find the column of eret_we whose name starts with 'pf':
    pfname = next((col for col in eret_we.columns if col.startswith('pf')), None)

    eret_subset, data = to_wide(eret_we, pfname)

    portfolios = [col for col in eret_subset.columns if col.startswith('pf')]
    
    colnames = portfolios + factors

    if quadriweekly:
        pseudo_monthly = rp.to_quadriweekly(data, colnames)

        # for col in pseudo_monthly.columns:
        #     if col != 'date':
        #         pseudo_monthly[col] = pseudo_monthly[col].apply(log_return_to_percentage)
    else:
        pseudo_monthly = data
        
    mod = LinearFactorModelGMM.from_formula(formula, pseudo_monthly, portfolios=pseudo_monthly[portfolios])
    results = mod.fit()
    return results

def log_return_to_percentage(log_return):
    return (np.exp(log_return) - 1)


def to_wide(eret_we, pfname):
    eret_wide = eret_we[['yw', pfname, 'eretw']]
    eret_we_agg = (eret_we
        .groupby('yw')
        .mean())
    
    eret_wide = eret_wide.pivot_table(index='yw', columns=pfname, values='eretw')
    eret_wide = eret_wide.add_prefix('pf')
    eret_wide = eret_wide.reset_index()
    eret_wide = viz.preprocess_eret_we(eret_wide)
    eret_wide['date'] = viz.convert_yw_to_date(eret_wide['yw'])
    # Add a new column called date:
    data = pd.concat([eret_wide.set_index('yw'), eret_we_agg[['Mkt.RF', 'SMB', 'HML', 'HKR']]], axis=1)
    return eret_wide, data