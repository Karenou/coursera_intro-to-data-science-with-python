import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.options.mode.chained_assignment = None

# read energy, GDP and ScimEn data
def answer_one():
    # remove header and footer
    energy = pd.read_excel('Energy Indicators.xls', skipfooter= 38 , skiprows= 17)
    energy = energy.iloc[:,2:]
    # rename column and index
    energy.columns = ['Country','Energy Supply', 'Energy Supply per Capita','% Renewable']
    energy['Country'] = energy['Country'].replace({"Republic of Korea": "South Korea",
                                                   'Australia1': 'Australia',
                                                   'United States of America20': 'United States',
                                                   'Bolivia (Plurinational State of)': 'Bolivia',
                                                   'China2': 'China',
                                                   'Region3': 'Region',
                                                   'China, Macao Special Administrative Region4': 'Macao',
                                                   'Denmark5': 'Denmark',
                                                   'Falkland Islands (Malvinas)': 'Falkland Islands',
                                                   'United Kingdom of Great Britain and Northern Ireland19': 'United Kingdom',
                                                   'China, Hong Kong Special Administrative Region': 'Hong Kong',
                                                   'France6':'France',
                                                   'Greenland7':'Greenland',
                                                   'Indonesia8':'Indonesia',
                                                   'Iran (Islamic Republic of)':'Iran',
                                                   'Italy9':'Italy',
                                                   'Japan10':'Japan',
                                                   'Kuwait11':'Kuwait',
                                                   'Micronesia (Federated States of)':'Micronesia',
                                                   'Netherlands12':'Netherlands',
                                                   'Portugal13':'Portugal',
                                                   'Venezuela (Bolivarian Republic of)':'Venezuela',
                                                  'Spain16':'Spain'})
    # convert Energy Supply to gigajoules (there are 1,000,000 gigajoules in a petajoule)
    energy['Energy Supply'] *= 1000000
    energy['Energy Supply per Capita'] = energy['Energy Supply per Capita'].replace('...', np.NaN).apply(pd.to_numeric)

    GDP = pd.read_csv('world_bank.csv',skiprows=4)
    # GDP.rename(columns={'Country Name': 'Country'}) why no effect
    GDP['Country Name'] = GDP['Country Name'].replace({'Korea, Rep.': 'South Korea',
                                                       'Iran, Islamic Rep.': 'Iran',
                                                       'Hong Kong SAR, China': 'Hong Kong'})
    GDP = GDP[['Country Name', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    GDP.columns = ['Country', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

    ScimEn = pd.read_excel('scimagojr-3.xlsx')
    ScimEn.rename(columns= {'country': 'Country'})

    df = pd.merge(ScimEn, energy, left_on= 'Country', right_on= 'Country', how= 'inner')
    df = pd.merge(df, GDP, left_on='Country', right_on = 'Country', how='inner')
    df = df.set_index('Country')
    # df[df['Rank'] <= 15].to_csv('merge data.csv', index = True)
    return df[df['Rank'] <= 15]


# all data minus top 15
def answer_two():
    Top15= answer_one()
    energy = pd.read_excel('Energy Indicators.xls', skipfooter= 38 , skiprows= 17)
    energy = energy.iloc[:,2:]
    # rename column and index
    energy.columns = ['Country','Energy Supply', 'Energy Supply per Capita','% Renewable']
    energy['Country'] = energy['Country'].replace({"Republic of Korea": "South Korea",
                                                   'Australia1': 'Australia',
                                                   'United States of America20': 'United States',
                                                   'Bolivia (Plurinational State of)': 'Bolivia',
                                                   'China2': 'China',
                                                   'Region3': 'Region',
                                                   'China, Macao Special Administrative Region4': 'Macao',
                                                   'Denmark5': 'Denmark',
                                                   'Falkland Islands (Malvinas)': 'Falkland Islands',
                                                   'United Kingdom of Great Britain and Northern Ireland19': 'United Kingdom',
                                                   'China, Hong Kong Special Administrative Region': 'Hong Kong',
                                                   'France6':'France',
                                                   'Greenland7':'Greenland',
                                                   'Indonesia8':'Indonesia',
                                                   'Iran (Islamic Republic of)':'Iran',
                                                   'Italy9':'Italy',
                                                   'Japan10':'Japan',
                                                   'Kuwait11':'Kuwait',
                                                   'Micronesia (Federated States of)':'Micronesia',
                                                   'Netherlands12':'Netherlands',
                                                   'Portugal13':'Portugal',
                                                   'Venezuela (Bolivarian Republic of)':'Venezuela',
                                                  'Spain16':'Spain'})
    # convert Energy Supply to gigajoules (there are 1,000,000 gigajoules in a petajoule)
    energy['Energy Supply'] *= 1000000
    energy['Energy Supply per Capita'] = energy['Energy Supply per Capita'].replace('...', np.NaN).apply(pd.to_numeric)

    GDP = pd.read_csv('world_bank.csv',skiprows=4)
    # GDP.rename(columns={'Country Name': 'Country'}) why no effect
    GDP['Country Name'] = GDP['Country Name'].replace({'Korea, Rep.': 'South Korea',
                                                       'Iran, Islamic Rep.': 'Iran',
                                                       'Hong Kong SAR, China': 'Hong Kong'})
    GDP = GDP[['Country Name', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    GDP.columns = ['Country', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']

    ScimEn = pd.read_excel('scimagojr-3.xlsx')
    ScimEn.rename(columns= {'country': 'Country'})

    full_df = pd.merge(GDP, ScimEn, left_on='Country', right_on='Country', how='outer')
    full_df = pd.merge(full_df, energy, left_on='Country', right_on='Country', how='outer')

    df = pd.merge(ScimEn, energy, left_on= 'Country', right_on= 'Country', how= 'inner')
    df = pd.merge(df, GDP, left_on='Country', right_on = 'Country', how='inner')

    return len(full_df) - len(df)

# average GDP in the past 10 years for each country
def answer_three():
    full_df, df = answer_one()
    GDP = df.iloc[:, 10:].mean(axis = 1).tolist()
    avgGDP = pd.Series(GDP, index= df.index, )
    avgGDP.sort_values(ascending=False, inplace=True)
    return avgGDP

# GDP change for ethe country with the sixth largest average GDP
def answer_four():
    columns = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    full_df, df = answer_one()
    avgGDP = answer_three()
    country = avgGDP.index[5]
    maxGDP = df.loc[country, columns].max()
    minGDP = df.loc[country, columns].min()
    return maxGDP - minGDP


# find the average energy supply per capita for all top 15 countries
def answer_five():
    full_df, df = answer_one()
    return df.loc[:, 'Energy Supply per Capita'].mean(axis = 0)


# find the country with the largest % renewable
def answer_six():
    full_df, df = answer_one()
    country = df[df['% Renewable'] == df.loc[:,'% Renewable'].max(axis= 0)].index
    return df.loc[country, '% Renewable']

# calculate self-citation ratio and find the country with largest value
def answer_seven():
    full_df, df = answer_one()
    df['Self Citation Ratio'] = df.apply(lambda row: row['Self-citations'] / row['Citations'], axis=1)
    # df['Self Citation Ratio'] = df['Self-citations']/ df['Citations']
    country = df[df['Self Citation Ratio'] == df['Self Citation Ratio'].max(axis = 0)].index
    return df.loc[country, 'Self Citation Ratio']


# calculate estimated population, find the country with the third largest population
def answer_eight():
    full_df, df = answer_one()
    df['PopEst'] = df.apply(lambda row: row['Energy Supply'] / row['Energy Supply per Capita'], axis=1)
    # df['PopEst'] = df['Energy Supply']/ df['Energy Supply per Capita']
    sort_poplation = df.sort_values(by= 'PopEst', ascending= False).head(3).index.tolist()
    return df, sort_poplation[2]

# plot the scatter plot between citable docs per capita adn energy supply per capita
def plot9():
    Top15, a = answer_eight()
    Top15['Citable docs per Capita'] = Top15.apply(lambda row: row['Citable documents'] / row['PopEst'], axis=1)
    Top15['Citable docs per Capita'].astype('float32')
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])
    plt.show()
    return


# calculate the correlation between citable docs per capita and energy supply per capita, make sure both datatypes are float
def answer_nine():
    # plot9()
    Top15 = answer_one()
    Top15['PopEst'] = Top15.apply(lambda row: row['Energy Supply'] / row['Energy Supply per Capita'], axis=1)
    Top15['Citable docs per Capita'] = Top15.apply(lambda row: row['Citable documents'] / row['PopEst'], axis=1)
    return Top15['Citable docs per Capita'].astype('float64').corr(Top15['Energy Supply per Capita'].astype('float64'))


# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15,
# and a 0 if the country's % Renewable value is below the median.
def answer_ten():
    Top15, a = answer_nine()
    mean_renewable = Top15['% Renewable'].mean(axis = 0)
    HighRenew = pd.Series(Top15['% Renewable'].apply(lambda x: 1 if x > mean_renewable else 0), index= Top15.index)
    return HighRenew.sort_index(ascending=True)


# group by continent, calculate the number of countries, size, sum, mean and std of estimated population for each continent
def answer_eleven():
    ContinentDict = {'China': 'Asia',
                     'United States': 'North America',
                     'Japan': 'Asia',
                     'United Kingdom': 'Europe',
                     'Russian Federation': 'Europe',
                     'Canada': 'North America',
                     'Germany': 'Europe',
                     'India': 'Asia',
                     'France': 'Europe',
                     'South Korea': 'Asia',
                     'Italy': 'Europe',
                     'Spain': 'Europe',
                     'Iran': 'Asia',
                     'Australia': 'Australia',
                     'Brazil': 'South America'}
    Top15 = answer_one()
    Top15['PopEst'] = Top15.apply(lambda row: row['Energy Supply'] / row['Energy Supply per Capita'], axis=1)
    Top15['Continent'] = pd.Series(ContinentDict)
    Top15.reset_index(inplace=True)
    # continent = Top15.groupby('Continent').agg({'Country': 'count', 'PopEst': ['sum', 'mean', 'std']})
    continent = Top15.set_index('Continent').groupby(level=0)['PopEst'].agg(
        {'size': np.size, 'sum': np.sum, 'mean': np.mean, 'std': np.std})
    continent.columns = ['size','sum','mean','std']
    continent.fillna(value=0, inplace = True)
    return continent

# cut the % renewable into 5 bins and
def answer_twelve():
    Top15, a = answer_eleven()
    renewBin = pd.cut(Top15['% Renewable'], 5)
    return Top15.groupby(['Continent', renewBin]).agg({'Country': 'count'})


# format the population by thousand commas
def answer_thirteen():
    Top15, a = answer_nine()
    Top15['PopEst'] = Top15.apply(lambda  x: "{:,}".format(x['PopEst']), axis = 1)
    return Top15['PopEst']


def plot_optional():
    a, Top15 = answer_one()
    # different color for different countries
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter',
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'],
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6])

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    plt.show()
    print("This is an example of a visualization that can be created to help understand the data. \
This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' \
2014 GDP, and the color corresponds to the continent.")
    return


print(answer_two())
