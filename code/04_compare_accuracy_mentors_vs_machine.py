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
from statsmodels.stats.contingency_tables import Table
from statsmodels.stats.contingency_tables import StratifiedTable
from scipy.stats import norm

pd.set_option('display.max_rows', 100)

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

Likert_order = [
    "Definitely not",
    "Probably not",
    "Might or might not",
    "Probably yes",
    "Definitely yes",
]

# +

# 1) Build a K x 2 table with columns ordered as [attended, missed] = [0, 1]
ct = pd.crosstab(
    pd.Categorical(dfs['recent_next_appoint'], categories=Likert_order, ordered=True),
    pd.Categorical(dfs['miss_first'], categories=[0, 1], ordered=True)
).to_numpy()
# -

ct

# 2) Run the linear-by-linear / Cochran–Armitage trend test
#    IMPORTANT: disable zero shifting so counts aren’t altered when there are zeros
tbl = Table(ct, shift_zeros=False)
res = tbl.test_ordinal_association()  # two-sided by default

print("Z (two-sided):", res.zscore)
print("p (two-sided):", res.pvalue)

# 3) Get a one-sided p-value for the hypothesized *increasing* trend
#    (Higher Likert -> higher miss risk). With columns [0, 1], a positive Z means "increasing".
p_one_sided = 1 - norm.cdf(res.zscore)
print("p (one-sided, increasing):", p_one_sided)


# +
def _build_counts(df, likert_col, y_col, order):
    """Returns (x, n, scores, groups) for the CA test.
       x[k] = #misses in category k; n[k] = total in k; scores = 1..K;
       groups[i] = category index for person i (length N)."""
    d = df[df[likert_col].isin(order)].copy()
    cat = pd.Categorical(d[likert_col], categories=order, ordered=True)
    d = d.assign(_cat=cat).sort_values('_cat')
    # counts per category
    grp = d.groupby('_cat', observed=True)[y_col]
    x = grp.sum().reindex(order).fillna(0).astype(int).to_numpy()
    n = grp.count().reindex(order).fillna(0).astype(int).to_numpy()
    # per-person group indices (0..K-1)
    groups = d['_cat'].cat.codes.to_numpy()
    # per-person outcome vector (0/1), not used in permutation (we only need totals)
    scores = np.arange(1, len(order) + 1, dtype=float)
    return x, n, scores, groups

def _ca_T_stat(x, n, scores):
    """Linear-by-linear (CA) numerator T = sum s_k * (x_k - n_k * X/N)."""
    N = n.sum()
    X = x.sum()
    p = X / N
    return float(np.sum(scores * (x - n * p)))

def _ca_asymptotic_Z(x, n, scores):
    """Asymptotic Z for reference (can be NaN if var=0)."""
    N = n.sum()
    if N == 0:
        return np.nan
    X = x.sum()
    p = X / N
    sbar = np.sum(n * scores) / N
    T = np.sum(scores * (x - n * p))
    varT = p * (1 - p) * np.sum(n * (scores - sbar) ** 2)
    return float(T / np.sqrt(varT)) if varT > 0 else np.nan

def cochran_armitage_permutation(
    df, likert_col='likert', y_col='missed',
    order=Likert_order, alternative='increasing',
    n_perm=100_000, random_state=17
):
    """
    Monte-Carlo permutation CA trend test (conditional on margins).
    alternative: 'increasing', 'decreasing', or 'two-sided'
    Returns dict with T_obs, p_perm, (optional) Z_asym and p_asym.
    """
    rng = np.random.default_rng(random_state)
    x, n, scores, groups = _build_counts(df, likert_col, y_col, order)
    N = n.sum()
    X = x.sum()
    if N == 0 or X == 0 or X == N:
        # degenerate cases: no data or all same outcome
        return {'T_obs': 0.0, 'p_perm': 1.0, 'Z_asym': np.nan, 'p_asym': 1.0}

    # observed statistic
    T_obs = _ca_T_stat(x, n, scores)

    # permutation: draw X indices as "misses" without replacement, count per group
    extremal = 0
    for _ in range(n_perm):
        miss_idx = rng.choice(N, size=X, replace=False)
        miss_counts = np.bincount(groups[miss_idx], minlength=len(n))
        T_perm = _ca_T_stat(miss_counts, n, scores)
        if alternative == 'increasing':
            extremal += (T_perm >= T_obs)
        elif alternative == 'decreasing':
            extremal += (T_perm <= T_obs)
        else:  # two-sided
            extremal += (abs(T_perm) >= abs(T_obs))

    # add-one smoothing (unbiased MC p)
    p_perm = (extremal + 1) / (n_perm + 1)

    # optional asymptotic reference
    Z = _ca_asymptotic_Z(x, n, scores)
    if alternative == 'increasing':
        p_asym = 1 - norm.cdf(Z)
    elif alternative == 'decreasing':
        p_asym = norm.cdf(Z)
    else:
        p_asym = 2 * min(norm.cdf(Z), 1 - norm.cdf(Z))

    return {'T_obs': float(T_obs), 'p_perm': float(p_perm), 'Z_asym': Z, 'p_asym': float(p_asym)}

# ---------- example usage ----------
# result = cochran_armitage_permutation(df, 'likert', 'missed',
#                                       order=LIKERT_ORDER,
#                                       alternative='increasing',
#                                       n_perm=100_000, random_state=1)
# print(result)


# -

cochran_armitage_permutation(dfs, 'recent_next_appoint', 'miss_first',
    order=Likert_order,
    alternative='increasing',
    n_perm=100_000, random_state=1)

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

# Note that 7 of the 13 of the mentor mothers do not have any clients that missed 
pick_comp2.groupby('mentor_username').miss_first.agg(['count','mean'])

# +
# 1) Clean and coerce to binary 0/1
dfc = pick_comp2[['miss_first','picked','mentor_username']].copy()

# If values aren’t already 0/1, coerce them deterministically:
# treat any truthy/positive as 1, everything else as 0
dfc['miss_first'] = (dfc['miss_first'].astype(float) > 0).astype(int)
dfc['picked']     = (dfc['picked'].astype(float) > 0).astype(int)

# Drop rows with missing key fields
dfc = dfc.dropna(subset=['miss_first','picked','mentor_username'])

# 2) Diagnose strata that break 2×2
def strata_diagnostics(g):
    return pd.Series({
        'n': len(g),
        'has_miss0': (g['miss_first'] == 0).any(),
        'has_miss1': (g['miss_first'] == 1).any(),
        'has_pick0': (g['picked']     == 0).any(),
        'has_pick1': (g['picked']     == 1).any(),
        'levels_miss': sorted(g['miss_first'].unique().tolist()),
        'levels_pick': sorted(g['picked'].unique().tolist()),
    })

diag = dfc.groupby('mentor_username', as_index=True).apply(strata_diagnostics, include_groups=False)

# Strata must have both levels for BOTH variables to contribute to CMH
good = diag.index[diag[['has_miss0','has_miss1','has_pick0','has_pick1']].all(axis=1)]
bad  = diag.index.difference(good)

# (optional) Inspect what’s being dropped
print("Dropping non-informative strata (no variation):", list(bad))

df_good = dfc[dfc['mentor_username'].isin(good)].copy()

# 3A) Easiest: run CMH directly on the filtered data
st = StratifiedTable.from_data(var1="miss_first",
                               var2="picked",
                               strata="mentor_username",
                               data=df_good)

cmh = st.test_null_odds()   # CMH χ²
or_hat = st.oddsratio_pooled
ci_lo, ci_hi = st.oddsratio_pooled_confint(method="logit")  # stable with sparse cells

print(f"CMH χ² = {cmh.statistic:.3f}, p = {cmh.pvalue:.4g}")
print(f"Common OR = {or_hat:.3f} (95% CI {ci_lo:.3f}, {ci_hi:.3f})")

# (optional) Check homogeneity across CHWs
bd = st.test_equal_odds()  # Breslow–Day test
print(f"Breslow–Day χ² = {bd.statistic:.3f}, p = {bd.pvalue:.4g}")
# -

stats.f_oneway(
    pick_comp2[pick_comp2.picked==1].miss_first,
    pick_comp2[pick_comp2.picked==0].miss_first)

pick_comp.mentor_username.nunique()

# Now compare to top 13 picked by ML

picked_by_ml = set(pick_comp2.loc[pick_comp2.MissFirst_Prediction_prob_brf.nlargest(13).index].case_id.to_list())

pick_comp2['picked_by_ml'] = np.where(pick_comp2.case_id.isin(picked_by_ml),1,0)

pick_comp2.groupby('picked_by_ml').miss_first.agg(['mean','count','sum'])
