import pandas as pd
import numpy as np
import  scipy
from scipy import stats

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ],
    columns=["State", "RegionName"]  )

    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''

    data = []
    state_town = []

    with open('university_towns.txt') as f:
        for line in f:
            data.append(line)

# indexing for special characters
    for line in data:
        if 'edit' in line:
            state = line[:line.index('[')]
        elif '(' in line:
            town = line[: line.index('(')-1]
            state_town.append([state,town])
    state_town = pd.DataFrame(state_town, columns=['State','RegionName'])
    return state_town

print(get_list_of_university_towns())

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a
    string value in a format such as 2005q3'''
    quarterGDP = pd.read_excel('gdplev.xls',skiprows=7)
    quarterGDP = quarterGDP.iloc[212:,[4,6]]
    quarterGDP.columns = ['quarter', 'GDP']
    quarterGDP.set_index('quarter',inplace=True)

    # find consecqutive growth or decline
    qoq_GDP = np.sign(quarterGDP.pct_change(periods=1))
    consequtive_qoq = qoq_GDP.diff(periods=1, axis=0).fillna(0)
    recession_begin = False
    recession_end = False
    recession_time = ''
    for i in range(0,len(consequtive_qoq)-1):
        if (consequtive_qoq.iloc[i]['GDP'] == -2.0) and (consequtive_qoq.iloc[i+1]['GDP'] == 0.0):
            recession_begin = True
            recession_time = consequtive_qoq.index[i]
        elif (consequtive_qoq.iloc[i]['GDP'] == 2.0) and (consequtive_qoq.iloc[i+1]['GDP'] == 0.0):
            recession_end = True
        # the case when GDP growth appears before decline
        if recession_time != '' :
            if (not recession_begin and recession_end):
                recession_end = False
                recession_time = ''
    return recession_time

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a
    string value in a format such as 2005q3'''
    quarterGDP = pd.read_excel('gdplev.xls',skiprows=7)
    quarterGDP = quarterGDP.iloc[212:,[4,6]]
    quarterGDP.columns = ['quarter', 'GDP']
    quarterGDP.set_index('quarter',inplace=True)

    # find consecqutive growth or decline
    qoq_GDP = np.sign(quarterGDP.pct_change(periods=1))
    consequtive_qoq = qoq_GDP.diff(periods=1, axis=0).fillna(0)
    recession_begin = False
    recession_end = False
    recession_end_time = ''
    for i in range(0,len(consequtive_qoq)-1):
        if (consequtive_qoq.iloc[i]['GDP'] == -2.0) and (consequtive_qoq.iloc[i+1]['GDP'] == 0.0):
            recession_begin = True
        elif (consequtive_qoq.iloc[i]['GDP'] == 2.0) and (consequtive_qoq.iloc[i+1]['GDP'] == 0.0):
            recession_end = True
            recession_end_time = consequtive_qoq.index[i+1]
        # the case when GDP growth appears before decline
        if recession_end_time != '' :
            if (not recession_begin and recession_end):
                recession_end = False
                recession_end_time = ''
            elif recession_begin and recession_end:
                break
    return recession_end_time


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a
    string value in a format such as 2005q3'''
    quarterGDP = pd.read_excel('gdplev.xls',skiprows=7)
    quarterGDP = quarterGDP.iloc[212:,[4,6]]
    quarterGDP.columns = ['quarter', 'GDP']
    quarterGDP.set_index('quarter',inplace=True)

    # find consecqutive growth or decline
    qoq_GDP = np.sign(quarterGDP.pct_change(periods=1))
    consequtive_qoq = qoq_GDP.diff(periods=1, axis=0).fillna(0)
    recession_begin = False
    recession_end = False
    recession_bottom = ''
    for i in range(0,len(consequtive_qoq)-1):
        if (consequtive_qoq.iloc[i]['GDP'] == -2.0) and (consequtive_qoq.iloc[i+1]['GDP'] == 0.0):
            recession_begin = True
        elif (consequtive_qoq.iloc[i]['GDP'] == 2.0) and (consequtive_qoq.iloc[i+1]['GDP'] == 0.0):
            recession_end = True
        # the case when GDP growth appears before decline
        if not recession_begin and recession_end:
            recession_end = False
        # the case when still in recession
        elif recession_begin and not recession_end:
            if (consequtive_qoq.iloc[i]['GDP'] == 0.0) and (consequtive_qoq.iloc[i+1]['GDP'] == 2.0):
                recession_bottom = consequtive_qoq.index[i]
    return recession_bottom


# convert abbrevation of state name to full name
def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    housing  = pd.read_csv('City_Zhvi_AllHomes.csv',header=0)
    housing = pd.concat([housing.iloc[:,1:3], housing.iloc[:,51:]], axis=1)
    for index, row in housing.iterrows():
        housing.at[index, 'State'] = states[row[1]]

    year = 2000
    quarter = 1
    i = 2
    num_quarter = 1
    while num_quarter < 68 :
        if str(year) not in housing.columns[i]:
            year += 1
            quarter = 1
        time = str(year) + 'q' + str(quarter)
        if time == '2016q3':
            housing[time] = housing.iloc[:, i:i + 2].mean(axis=1)
        else:
            housing[time] = housing.iloc[:,i:i+3].mean(axis = 1)
        quarter += 1
        i += 3
        num_quarter += 1

    housing = pd.concat([housing.iloc[:, :2], housing.iloc[:, 202:]], axis=1)
    housing.set_index(['State', 'RegionName'], inplace = True)
    return housing


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''

    # print(get_recession_start())
    quarter_before_recession = '2008q2' # quarter before recession start
    quarter_recession_bottom = get_recession_bottom()
    housing  = convert_housing_data_to_quarters()
    price_ratio = pd.DataFrame(housing[quarter_before_recession] / housing[quarter_recession_bottom], columns=['price_ratio'])

    # separate university and non-university towns
    state_town = get_list_of_university_towns()
    state_town = state_town.to_records(index=False).tolist()
    uni = price_ratio.loc[price_ratio.index.isin(state_town)]
    non_uni = price_ratio.loc[~price_ratio.index.isin(state_town)]

    p = scipy.stats.ttest_ind(uni, non_uni,nan_policy='omit').pvalue
    if p < 0.01:
        difference = True
    else:
        difference = False
    if uni['price_ratio'].mean() > non_uni['price_ratio'].mean():
        better = 'non-university town'
    else:
        better = 'university town'

    return (difference, p, better)

