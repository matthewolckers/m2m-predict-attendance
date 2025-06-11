# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: m2m
#     language: python
#     name: python3
# ---

import pandas as pd
import datetime as dt
import numpy as np
from scipy import stats
import statsmodels.stats.proportion as proportion

pred = pd.read_csv("../data/comp_predictions.csv", dtype={'province':'category'})

survey = pd.read_csv("../data/survey.csv")

survey_recent_caseid = pd.read_csv("../data/survey_recent_caseid.csv")

survey_pick_caseid = pd.read_csv("../data/survey_pick_caseid.csv")

mm_recent = survey.merge(survey_recent_caseid, how="left",on="ResponseId")

df = mm_recent.merge(pred,how="left",on="case_id")

mm_pick = survey.merge(survey_pick_caseid, how="left",on="ResponseId")

df_pick = mm_pick.merge(pred,how="left",on="case_id")

pick_comp = pd.read_csv("../data/survey_pick_comparison_set.csv")

pick_comp = pick_comp.merge(pred,how="left",on="case_id")

# ## Miss next appointment

likert_scale =  ['Definitely yes',
                 'Probably yes',
                 'Might or might not',
                 'Probably not',
                 'Definitely not']

df.recent_next_appoint.value_counts().reindex(likert_scale)

df.recent_next_appoint.count()

#Share of mentor mothers that answered definitely or probably yes
((df[df.recent_next_appoint=="Definitely yes"].recent_next_appoint.count()+
df[df.recent_next_appoint=="Probably yes"].recent_next_appoint.count())/
df.recent_next_appoint.count())*100

# ### Merge survey and admin data

df['startedtime_admin'] = pd.to_datetime(df.startedtime_admin, format='%d/%m/%Y')

df['StartDate'] = pd.to_datetime(df.StartDate, format='%d/%m/%Y %H:%M')

df['days_diff'] = (df.startedtime_admin-df.StartDate).dt.days 

df.case_id.count()

dfs = df[(df.days_diff<=1)&(df.days_diff>=-30)&(df.miss_first.notnull())].copy()

dfs.case_id.count()

# ### Accuracy

pred.miss_first.agg(['count','mean'])

dfs.miss_first.agg(['count','mean'])

dfs.groupby('recent_next_appoint').miss_first.agg(['count','mean']).reindex(likert_scale)

stats.f_oneway(
    dfs[dfs.recent_next_appoint=="Definitely yes"].miss_first,
    dfs[dfs.recent_next_appoint=="Probably yes"].miss_first,
    dfs[dfs.recent_next_appoint=="Might or might not"].miss_first,
    dfs[dfs.recent_next_appoint=="Probably not"].miss_first,
    dfs[dfs.recent_next_appoint=="Definitely not"].miss_first
)

# ### Comparison to ML

dfs.loc[dfs.MissFirst_Prediction_prob_brf.nlargest(1).index].miss_first.agg(['count','mean'])

dfs[dfs.recent_next_appoint.isin(['Definitely yes'])].miss_first.agg(['count','mean'])

dfs.loc[dfs.MissFirst_Prediction_prob_brf.nlargest(3).index].miss_first.agg(['count','mean'])

dfs[dfs.recent_next_appoint.isin(['Definitely yes','Probably yes'])].miss_first.agg(['count','mean'])

dfs.loc[dfs.MissFirst_Prediction_prob_brf.nlargest(12).index].miss_first.agg(['count','mean'])

dfs[dfs.recent_next_appoint.isin(['Definitely yes','Probably yes','Might or might not'])].miss_first.agg(['count','mean'])

dfs.loc[dfs.MissFirst_Prediction_prob_brf.nlargest(31).index].miss_first.agg(['count','mean'])

dfs[dfs.recent_next_appoint.isin([
    'Definitely yes',
    'Probably yes',
    'Might or might not',
    'Probably not'])].miss_first.agg(['count','mean'])

dfs["Definitely_not_miss_next"] = dfs.recent_next_appoint.map({
    'Definitely not':1,
    'Probably not':0,
    'Might or might not':0,
    'Probably yes':0,
    'Definitely yes':0})

dfs.groupby("Definitely_not_miss_next").miss_first.agg(['count','mean'])

stats.ttest_ind(
    dfs[dfs.Definitely_not_miss_next==1].miss_first, 
    dfs[dfs.Definitely_not_miss_next==0].miss_first,
    equal_var=False, alternative='less')

# See information on the [hypothesis test for two independent proportions](https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.confint_proportions_2indep.html).

proportion.confint_proportions_2indep(
    count1=dfs[dfs.Definitely_not_miss_next==0].miss_first.sum(),
    nobs1=dfs[dfs.Definitely_not_miss_next==0].miss_first.count(),
    count2=dfs[dfs.Definitely_not_miss_next==1].miss_first.sum(),
    nobs2=dfs[dfs.Definitely_not_miss_next==1].miss_first.count(),
    compare='diff')

# ## Pick data

df_pick.mentor_username.nunique()

df_pick.miss_first.count()

df_pick['startedtime_admin'] = pd.to_datetime(df_pick.startedtime_admin, format='%d/%m/%Y')

df_pick['StartDate'] = pd.to_datetime(df_pick.StartDate, format='%d/%m/%Y %H:%M')

df_pick['days_diff'] = (df_pick.startedtime_admin-df_pick.StartDate).dt.days 

df_pick[(df_pick.days_diff<=0)&(df_pick.days_diff>=-7)].miss_first.count()

df_pick[(df_pick.days_diff<=0)&(df_pick.days_diff>=-7)].miss_first.mean()

df_pick.columns

# Rule to select comparison set:
# - Let x be number of clients in the admin date in the previous seven days up to the date of the survey.
# - Let y be the number of clients self reported by the mentor mother in the survey. 
# - If x >= y: Then take the y most recent clients up to the date of the survey.
# - If x < y: The take the x clients in the 7 day window 

picked = set(df_pick[(df_pick.days_diff<=0)&(df_pick.days_diff>=-7)].case_id.to_list())

pick_comp['picked'] = np.where(pick_comp.case_id.isin(picked),1,0)

no_data_for_picked = set(
    pick_comp[(pick_comp.miss_first.isnull())
    &(pick_comp.picked==1)].mentor_username.unique())

pick_comp2 = pick_comp[
    (~pick_comp.mentor_username.isin(no_data_for_picked))&
    (pick_comp.miss_first.notnull())]

pick_comp2.groupby('picked').miss_first.agg(['count','mean'])

stats.f_oneway(
    pick_comp2[pick_comp2.picked==1].miss_first,
    pick_comp2[pick_comp2.picked==0].miss_first)

pick_comp.mentor_username.nunique()

# Now compare to top 13 picked by ML

picked_by_ml = set(pick_comp2.loc[pick_comp2.MissFirst_Prediction_prob_brf.nlargest(13).index].case_id.to_list())

pick_comp2['picked_by_ml'] = np.where(pick_comp2.case_id.isin(picked_by_ml),1,0)

pick_comp.groupby('picked_by_ml').miss_first.agg(['mean','count','sum'])
