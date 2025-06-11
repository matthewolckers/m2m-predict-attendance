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

pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# ### Specify data types
#
# Specify datatypes for each feature. See [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html?highlight=basics#dtypes) for data types

new_vars_dtype = {
    'case_id':'string',
    'program':'category',
    'gender':'category',
    'age':'Int64',
    'status':'category',
    'country':'category',
    'province':'category',
    'facility':'category',
    'm2m_enrollment_date':'object',
    'has_partner':'category',
    'partner_status':'category',
    'is_this_women_pregnant_or_lactating':'category',
    'm2m_before':'category',
    'm2m_how_many_times':'Int64',
    'contact_method':'category',
    'agree_to_acfu':'category',
    'client_type':'category',
    'dob_known':'category',
    'm2m_community_client':'category',
    'edd_known':'category',
    'username':'string'}

an_vars_dtype = {
    'case_id':'string',
    'adherence.five_point':'category',
    'mentor_mother_notes':'string'}

pn_vars_dtype = {
    'case_id':'string',
    'adherence.five_point':'category',
    'mentor_mother_notes':'string'}

acfu_vars_dtype = {
    'case_id':'string',
    'acfu_calls.call_1.call_1_outcome':'category',
    'acfu_calls.call_2.call_2_outcome':'category',
    'acfu_calls.call_3.call_3_outcome':'category',
    'acfu_home_visit.home_visit1.home_visit1_no_cmm.home_visit1_outcome':'category',
    'acfu_home_visit.home_visit2.home_visit2_no_cmm.home_visit2_outcome':'category'}

exit_vars_dtype = {
    'case_id':'string',
    'close_reason':'category'}

new_vars_dates = [
    "started_time","completed_time","m2m_enrollment_date","date_of_conception",
    "next_visit_date","next_acfu_date","edd"]

an_vars_dates = ["started_time","completed_time","next_visit_date","next_acfu_date"]

pn_vars_dates = ["started_time","completed_time","next_visit_date","next_acfu_date"]

acfu_vars_dates = ["started_time","completed_time"]

exit_vars_dates = ["started_time","completed_time"]

new_vars = list(new_vars_dtype.keys()) + new_vars_dates
an_vars = list(an_vars_dtype.keys()) + an_vars_dates
pn_vars = list(pn_vars_dtype.keys()) + pn_vars_dates
acfu_vars = list(acfu_vars_dtype.keys()) + acfu_vars_dates
exit_vars = list(exit_vars_dtype.keys()) + exit_vars_dates

# ## Load data
#
# ### 2020-2022 data

print("Start loading data " + str(dt.datetime.now()) + "\n")
new20 = pd.read_csv("../data/new_2020-2022.csv", usecols=new_vars, dtype=new_vars_dtype, parse_dates=new_vars_dates, dayfirst=True)
print("New loaded at " + str(dt.datetime.now()) + "\n")
an20 = pd.read_csv("../data/an_2020-2022.csv", usecols=an_vars, dtype=an_vars_dtype, parse_dates=an_vars_dates, dayfirst=True)
print("AN loaded at " + str(dt.datetime.now()) + "\n")
pn20 = pd.read_csv("../data/pn_2020-2022.csv", usecols=pn_vars, dtype=pn_vars_dtype, parse_dates=pn_vars_dates, dayfirst=True)
print("PN loaded at " + str(dt.datetime.now()) + "\n")
acfu20 = pd.read_csv("../data/acfu_2020-2022.csv", usecols=acfu_vars, dtype=acfu_vars_dtype, parse_dates=acfu_vars_dates, dayfirst=True)
print("ACFU loaded at " + str(dt.datetime.now()) + "\n")
exit20 = pd.read_csv("../data/exit_2020-2022.csv", usecols=exit_vars, dtype=exit_vars_dtype, parse_dates=exit_vars_dates, dayfirst=True)
print("Exit loaded at " + str(dt.datetime.now()) + "\n")

for x in ['next_visit_date','next_acfu_date']:
    pn20[x] = pd.to_datetime(pn20[x], errors='coerce', dayfirst=True)

for df in [new20, an20, pn20, acfu20, exit20]:
    df["source"] = "2020-22 data"

# ### 2023 data

print("Start loading data " + str(dt.datetime.now()) + "\n")
new23 = pd.read_csv("../data/new_2023.csv", usecols=new_vars, dtype=new_vars_dtype, parse_dates=new_vars_dates, date_format="%d/%m/%y")
print("New loaded at " + str(dt.datetime.now()) + "\n")
an23 = pd.read_csv("../data/an_2023.csv", usecols=an_vars, dtype=an_vars_dtype, parse_dates=an_vars_dates, date_format="%d/%m/%y")
print("AN loaded at " + str(dt.datetime.now()) + "\n")
pn23 = pd.read_csv("../data/pn_2023.csv", usecols=pn_vars, dtype=pn_vars_dtype, parse_dates=pn_vars_dates, date_format="%d/%m/%y")
print("PN loaded at " + str(dt.datetime.now()) + "\n")
acfu23 = pd.read_csv("../data/acfu_2023.csv", usecols=acfu_vars, dtype=acfu_vars_dtype, parse_dates=acfu_vars_dates, date_format="%d/%m/%y")
print("ACFU loaded at " + str(dt.datetime.now()) + "\n")
exit23 = pd.read_csv("../data/exit_2023.csv", usecols=exit_vars, dtype=exit_vars_dtype, parse_dates=exit_vars_dates, date_format="%d/%m/%y")
print("Exit loaded at " + str(dt.datetime.now()) + "\n")

for x in new_vars_dates:
    new23[x] = pd.to_datetime(new23[x])
for x in an_vars_dates:
    an23[x] = pd.to_datetime(an23[x])
for x in pn_vars_dates:
    pn23[x] = pd.to_datetime(pn23[x])
for x in acfu_vars_dates:
    acfu23[x] = pd.to_datetime(acfu23[x])
for x in exit_vars_dates:
    exit23[x] = pd.to_datetime(exit23[x])

for df in [new23, an23, pn23, acfu23, exit23]:
    df["source"] = "2023 data"

# ### Combine datasets

new = pd.concat([new20,new23])

an = pd.concat([an20,an23])

pn = pd.concat([pn20,pn23])

acfu = pd.concat([acfu20,acfu23])

exit = pd.DataFrame()
exit = pd.concat([exit20,exit23])

# ## Data cleaning and variable creation

datasets = {"new":new,"an":an,"pn":pn,"acfu":acfu,"exit":exit}

for df_name in datasets.keys():
    print("-----------------------------------------")
    print(df_name)
    datasets[df_name].info(verbose=False, memory_usage="deep")
    datasets[df_name]["month"] = datasets[df_name].started_time.dt.strftime('%Y-%m')
    datasets[df_name]["dataset"] = df_name

new.month.value_counts().sort_index()

# Mentor mother survey in 2022:
#
# Month | Count
# --- | ---
# August |	7
# October |	81
# November |	115
# December |	1

start_train = "2021-01-01"
end_train = "2021-10-31"
start_test="2021-11-01"
end_test = "2022-01-31"
start_comp = "2022-08-01"
end_comp = "2022-12-31"

# Exclude clients who are not in the PMTCT program, are male, and are not pregant or lactacting.
#
# * Earlier versions of our analysis excluded clients that were HIV negative. But our survey data includes lots of clients that are HIV negative so we would throw out too much data if we had to use this restriction.

df = new[(new['program']=="PMTCT program")&(new["gender"]!="male")&
              (new["is_this_women_pregnant_or_lactating"]!="no")&
              (new["started_time"]>=start_train)&(new["started_time"]<=end_comp)].copy()

# In Active Client Follow Up
df["In_ACFU"] = df["case_id"].isin(acfu["case_id"]).astype(int)

# app is short for appointments
app = pd.concat([new[['case_id','dataset','started_time','next_visit_date']],
                            an[['case_id','dataset','started_time','next_visit_date']],
                            pn[['case_id','dataset','started_time','next_visit_date']]])

app.sort_values(['case_id','started_time'],inplace=True)

clients = pd.DataFrame()

clients["app_count"] = app.groupby('case_id').started_time.nunique()
clients["signup_date"] = new.groupby('case_id').started_time.max()
clients["most_recent_attendance"] = app.groupby('case_id').started_time.max()
clients["most_recent_app_scheduled"] = app.groupby('case_id').next_visit_date.max()

clients["missed_most_recent"] = np.where((clients.most_recent_attendance + dt.timedelta(days=14) < clients.most_recent_app_scheduled ),1,0)
# There is a break in our data between data dumps where we cannot observe if a client missed
clients["missed_most_recent"] = np.where((clients.most_recent_app_scheduled>"2022-4-30")&(clients.most_recent_app_scheduled<"2022-08-01"),0,clients.missed_most_recent)
# If appointment scheduled after our data ends, we cannot observe whether the client missed
clients["missed_most_recent"] = np.where(clients.most_recent_app_scheduled>"2023-05-31",0,clients.missed_most_recent)

clients["most_recent_app_scheduled_within_90_days"] = np.where(clients.most_recent_app_scheduled - clients.signup_date < dt.timedelta(days=90), 1, 0)
clients["missed_most_recent_within_90_days"] = clients.missed_most_recent * clients.most_recent_app_scheduled_within_90_days
#clients["5pointadherence"]=an.groupby('case_id')['adherence.five_point']
clients["first_an"] = an.groupby('case_id').started_time.min()
clients["first_pn"] = pn.groupby('case_id').started_time.min()
clients["days_between_anpn"] = np.fmax((clients.signup_date - clients.first_an).dt.days,(clients.signup_date - clients.first_pn).dt.days)

# Filtrations, such as .nth(0) don't use the groupby object as the index so you have to set the index as what you want returned
app.set_index('case_id',inplace=True) 
exit.set_index('case_id',inplace=True) 

# Note that nth starts at 0
clients["scheduled_0"] = app[app.dataset!="new"].groupby('case_id').nth(0).next_visit_date 
clients["visit_0"] = app[app.dataset!="new"].groupby('case_id').nth(0).started_time
clients["visit_1"] = app[app.dataset!="new"].groupby('case_id').nth(1).started_time
clients["visit_2"] = app[app.dataset!="new"].groupby('case_id').nth(2).started_time
clients["visit_3"] = app[app.dataset!="new"].groupby('case_id').nth(3).started_time
clients["diff_0_1"] = (clients.visit_1 - clients.scheduled_0).dt.days
clients["diff_0_2"] = (clients.visit_2 - clients.scheduled_0).dt.days
clients["diff_0_3"] = (clients.visit_3 - clients.scheduled_0).dt.days
clients["exit_0"] = exit.groupby("case_id").nth(0).started_time
clients["exit_1"] = exit.groupby("case_id").nth(1).started_time
clients["exit_2"] = exit.groupby("case_id").nth(2).started_time
clients["exit_0_reason"] = exit.groupby("case_id").nth(0).close_reason
clients["exit_1_reason"] = exit.groupby("case_id").nth(1).close_reason
clients["exit_2_reason"] = exit.groupby("case_id").nth(2).close_reason
clients["exit_0_near_first"] = np.where((clients.exit_0>clients.visit_0)&
                                        (clients.exit_0<clients.scheduled_0+dt.timedelta(days=30)),"yes","no")
clients["exit_1_near_first"] = np.where((clients.exit_1>clients.visit_0)&
                                        (clients.exit_1<clients.scheduled_0+dt.timedelta(days=30)),"yes","no")
clients["exit_2_near_first"] = np.where((clients.exit_2>clients.visit_0)&
                                        (clients.exit_2<clients.scheduled_0+dt.timedelta(days=30)),"yes","no")

# +
clients["miss_first"] = np.where((abs(clients.diff_0_1)<=7)|(abs(clients.diff_0_2)<=7)|(abs(clients.diff_0_3)<=7),0,1)

# mark rows as null when there is no scheduled appointment
clients["miss_first"] = np.where(clients.scheduled_0.isnull(),np.nan,clients.miss_first)

# if the client had an exit recorded near the scheduled appointment, mark as null
clients["miss_first"] = np.where((clients.exit_0_near_first=="yes")|
                                 (clients.exit_1_near_first=="yes")|
                                 (clients.exit_2_near_first=="yes"),np.nan,clients.miss_first)
# -

clients[["miss_first","missed_most_recent","missed_most_recent_within_90_days"]].corr()

clients.reset_index(inplace=True)

sum(clients.duplicated(subset="case_id"))

df = df.merge(clients, on="case_id", how='left')

anpn=pd.concat([an,pn])
anpn=anpn.sort_values(['started_time']).drop_duplicates(subset=['case_id'], keep='first')
anpn=anpn.drop(['mentor_mother_notes', 'next_visit_date', 'started_time', 'month', 'dataset', 'next_acfu_date', 'completed_time'],axis=1)
anpnfeatures = pd.DataFrame()
anpnfeatures['case_id']=clients["case_id"]
anpnfeatures["days_between_anpn"]=clients["days_between_anpn"]
anpnfeatures=anpnfeatures.merge(anpn, on="case_id", how='left')
#anpnfeatures=anpnfeatures[anpnfeatures['days_between_anpn']<0]
anpnfeatures=anpnfeatures[(anpnfeatures['days_between_anpn']==0) | (anpnfeatures['days_between_anpn']==-1)] #Date of latest new.csv entry = date of earliest an/pn entry
anpnfeatures=anpnfeatures[anpnfeatures.isnull().sum(axis=1) < 3] #only include rows that have at least one adherence value
anpnfeatures=anpnfeatures.drop('days_between_anpn',axis=1)
anpnfeatures.rename(columns={'adherence.five_point': 'adherence.five_point', 'adherence.seven_day_recall': 'adherence.seven_day_recall', 'mother_hiv_retest.mother_hiv_retest_status_entered': 'mother_hiv_retest_status_entered'}, inplace=True)
df = df.merge(anpnfeatures, on="case_id", how='left')

df["days_from_due_date"]=np.where(df.edd.isnull(),np.nan,(df["edd"]-df["started_time"]).dt.days)

df['duration_to_complete_form'] = (df.completed_time - df.started_time).dt.total_seconds() / 60

selected_continous_features = ['age','duration_to_complete_form','num_contact_methods','days_from_due_date']

df['adherence.five_point'].value_counts()

df['low_or_moderate_adherence'] = np.where((df['adherence.five_point']=="moderate_adherence")|
                                           (df['adherence.five_point']=="low_adherence"),1,0)

df["contact_method_phone_call"]=df["contact_method"].str.contains("phone_call", regex=False, na=0).astype(int)
df["contact_method_sms"]=df["contact_method"].str.contains("sms", regex=False, na=0).astype(int)
df["contact_method_home_visit"]=df["contact_method"].str.contains("home_visit", regex=False, na=0).astype(int)
df["num_contact_methods"]=df["contact_method_phone_call"]+df["contact_method_sms"]+df["contact_method_home_visit"]

# Create indicators for certain features
df['has_partner_yes'] = (df.has_partner=="yes").astype('int')
df['dob_known_yes'] = (df.dob_known=="yes").astype('int')
df['edd_known_yes'] = (df.edd_known=="yes").astype('int')
df['m2m_community_client_yes'] = (df.m2m_community_client=="yes").astype('int')
df['m2m_before_yes'] = (df.m2m_before=="yes").astype('int')
df['agree_to_acfu_no'] = (df.agree_to_acfu=="no").astype('int')
df['client_type_PN'] = (df.client_type=="PN").astype('int')
df['own_status_positive'] = (df.status=="b").astype('int')
df['own_status_unknown'] = (df.status=="c").astype('int')
df['partner_status_positive'] = (df.partner_status=="b").astype('int')
df['partner_status_unknown'] = (df.partner_status=="c").astype('int')

# Note started time is not a target feature but it is convenient to include in this group
target_features = ["In_ACFU","miss_first","missed_most_recent_within_90_days","started_time","case_id"]

categorical_features = ['province','contact_method_phone_call','contact_method_sms','contact_method_home_visit',
                       'has_partner_yes','dob_known_yes','edd_known_yes','m2m_community_client_yes',
                       'm2m_before_yes','agree_to_acfu_no','client_type_PN',
                        'own_status_positive','own_status_unknown',
                        'partner_status_positive','partner_status_unknown','low_or_moderate_adherence']

df_2 = df[target_features+selected_continous_features+categorical_features].copy()

train = df_2[(df_2["started_time"]>=start_train)&(df_2["started_time"]<=end_train)].copy()

test = df_2[(df_2["started_time"]>=start_test)&(df_2["started_time"]<=end_test)].copy()

comp = df_2[(df_2["started_time"]>=start_comp)&(df_2["started_time"]<=end_comp)].copy()

train.to_csv("../data/train_v4.csv", index=False)

test.to_csv("../data/test_v4.csv", index=False)

comp.to_csv("../data/comp_v4.csv", index=False)
