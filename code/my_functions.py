import pandas as pd
import numpy as np

def load_data():
    '''
    Load the data used for training, testing, and comparison. 
    '''
    train = pd.read_csv("../data/train_v4.csv", dtype={'province':'category'})
    test = pd.read_csv("../data/test_v4.csv", dtype={'province':'category'})
    comp = pd.read_csv("../data/comp_v4.csv", dtype={'province':'category'})

    train.sort_values('started_time', inplace=True)
    test.sort_values('started_time', inplace=True)
    comp.sort_values('started_time', inplace=True)

    train = train[train.miss_first.notnull()].copy()
    test = test[test.miss_first.notnull()].copy()
    comp = comp[comp.miss_first.notnull()].copy()

    train['duration_to_complete_form'] = np.where(
        train.duration_to_complete_form>train.duration_to_complete_form.quantile(0.99),
        train.duration_to_complete_form.quantile(0.99),
        train.duration_to_complete_form)
    
    train['days_from_due_date'] = np.where(train.days_from_due_date.isnull(),-1000,train.days_from_due_date)
    test['days_from_due_date'] = np.where(test.days_from_due_date.isnull(),-1000,test.days_from_due_date)
    comp['days_from_due_date'] = np.where(comp.days_from_due_date.isnull(),-1000,comp.days_from_due_date)

    features = [
        'age', 'duration_to_complete_form', 'days_from_due_date',
        'low_or_moderate_adherence', 'contact_method_phone_call',
        'contact_method_sms', 'contact_method_home_visit', 'has_partner_yes',
        'dob_known_yes', 'edd_known_yes', 'm2m_community_client_yes',
        'm2m_before_yes', 'agree_to_acfu_no', 'client_type_PN',
        'own_status_positive','own_status_unknown',
        'partner_status_positive', 'partner_status_unknown']
    
    y_train_miss_first = train.miss_first.copy()
    X_train = train[features].copy()
    y_test_miss_first = test.miss_first.copy()
    X_test = test[features].copy()
    y_comp_miss_first = comp.miss_first.copy()
    X_comp = comp[features].copy()

    return train, test, comp, y_train_miss_first, X_train, y_test_miss_first, X_test, y_comp_miss_first, X_comp


def create_descriptive_stats_table():
    '''
    We have to load the training data again because we need to skip the filtering
    step where we set null values of days_from_due_date to -1000. A better approach
    would be to ensure that the data is loaded with the correct settings in the
    my_functions.py file.
    '''
    train = pd.read_csv("../data/train_v4.csv", dtype={'province':'category'})

    train = train[train.miss_first.notnull()].copy()

    train['duration_to_complete_form'] = np.where(
        train.duration_to_complete_form>train.duration_to_complete_form.quantile(0.99),
        train.duration_to_complete_form.quantile(0.99),
        train.duration_to_complete_form)

    table_header = r'''\begin{table} 
    \centering 
    \begin{threeparttable} 
    \begin{tabular}{lc} 
    \toprule 
    Feature & Mean \\
    \midrule 
    \underline{Continuous features} & \\ '''

    continuous_features = {
        'age':"Age in years",
        'duration_to_complete_form':"Duration to complete form in minutes",
        'days_from_due_date':"Days to expected due date"}

    indicator_header = r" \underline{Indicators (1 = Yes, 0 = No)} & \\"

    indicator_features = {
        'm2m_before_yes':"m2m client before",
        'client_type_PN':"Postnatal client",
        'edd_known_yes':"Know expected due date",
        'agree_to_acfu_no':"Did not agree to follow up contact",
        'contact_method_home_visit':"Can contact with home visit",
        'contact_method_sms':"Can contact with SMS",
        'contact_method_phone_call':"Can contact with phone call",
        'own_status_positive':"Own HIV status is positive",
        'own_status_unknown':"Own HIV status is unknown",
        'partner_status_positive':"Partner HIV status is positive",
        'partner_status_unknown':"Partner HIV status is unknown",
        'dob_known_yes':"Knows own date of birth",
        'low_or_moderate_adherence':"Low or moderate adherence to treatment"}


    table_footer = r'''\bottomrule
    \end{tabular}
    \begin{tablenotes}[flushleft]
    \item \small \emph{Notes:} We report the mean of the feature. For continuous features, we report the 
    standard deviation in parenthesis. The summary statistics are 
    calculated using the ''' + "{:,.0f}".format(train.miss_first.count()) + r''' clients included in the training dataset.
    The days to the expected due date is 
    calculated using only the ''' + "{:,.0f}".format(train.days_from_due_date.count()) + r''' antenatal clients that 
    know their due date.
    \end{tablenotes}
    \end{threeparttable}
    \caption{Selected Features}
    \label{table:features}
    \end{table}'''

    print(table_header)
    for key, item in continuous_features.items():
        print(r"\ \ " + item + " & " + "{:.2f}".format(train[key].mean()) + " (" + "{:.2f}".format(train[key].std()) + r") \\" )
    print(indicator_header)
    for key, item in indicator_features.items():
        print(r"\ \ " + item + " & " + "{:.2f}".format(train[key].mean()) + r" \\" )
    print(table_footer)


