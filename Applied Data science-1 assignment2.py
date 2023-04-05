# Import Python Packages
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('API_19_DS2_en_csv_v2_5361599.csv', skiprows=4)

# Function


def read_world_bank_data(filename, skip):
    ''' a function which takes a
lename as argument, reads a dataframe and
returns two dataframes: one with years as columns and one with
countries as columns.'''

    data = pd.read_csv(filename, skiprows=skip)

    df_years = data.set_index(['Indicator Name', 'Country Name']).drop(
        ['Country Code', 'Indicator Code'], axis=1)

    df_countries = data.set_index(['Country Name']).drop(
        ['Country Code', 'Indicator Code'], axis=1).T

    return df_years, df_countries


# Function Call
df_data = read_world_bank_data('API_19_DS2_en_csv_v2_5361599.csv', 4)

# Using Statastical Function for data manipulation
country_data = data.groupby(['Country Name', 'Indicator Name']).sum()
country_data = data.groupby(['Country Name', 'Indicator Name']).mean()
indicator = country_data.iloc[country_data.index.get_level_values
                              ('Indicator Name') == 'Co2 emission(kt)\
                          (% of total final energy consumption)']
indicator.loc[:, '2005':'2010'].mean()


# Fetching the years & country from dr_data tuple..
years = df_data[0]
country = df_data[1]

# slicing the years
renewable_eng = years.loc['Renewable energy consumption\
                          (% of total final energy consumption)',
                          '2005': '2010']

print(renewable_eng)

# Droping the NaN values
renewable_eng = renewable_eng.dropna(axis=0)

# Slicing the indicator with some countries..
renewable_eng = renewable_eng.loc[[
    'Belize', 'Colombia', 'Cyprus', 'Germany', 'Estonia'], '2005': '2010']


# Ploting the graph
plt.figure(figsize=(10, 25))
renewable_eng.plot.bar()
plt.title('Renewable energy consumption')
plt.xlabel('Country', fontsize=12)
plt.xticks(rotation=45)
plt.ylabel(
    'energy(% of total final energy)$\longrightarrow$', fontsize=12)
plt.legend(prop={'size': 10}, loc='center left',
           bbox_to_anchor=(1, .86), fontsize='large')

co2_emission = years.loc['CO2 emissions (kt)', '2005':'2010']


co2_emission.dropna(axis=0)

co2_emission = co2_emission.loc[[
    'Belize', 'Colombia', 'Cyprus', 'Germany', 'Estonia'], '2005':'2010']
print(co2_emission)

# plotting the Co2 emission graph.
plt.figure(figsize=(10, 25))
co2_emission.plot.bar()
plt.title('CO2 emissions (kt)')
plt.xlabel('Country', fontsize=12)
plt.xticks(rotation=45)
plt.ylabel('CO2 emissions (kt) $\longrightarrow$', fontsize=12)
plt.legend(prop={'size': 10}, loc='center left',
           bbox_to_anchor=(1, .86), fontsize='large')


# Slicing the Years
primary_enrol = years.loc['Primary completion rate, total\
                          (% of relevant age group)',
                          '2005':'2010']

primary_enrol = primary_enrol.dropna(axis=0)

primary_enrol = primary_enrol.loc[[
    'Belize', 'Colombia', 'Cyprus', 'Germany', 'Estonia'], '2005':'2010']


# Tranpose the dataframe
primary_enrol = primary_enrol.T


# Ploting the graph..
matplotlib.rc('figure', figsize=(8, 6))
primary_enrol.plot(label='Label', linestyle=":",
                   marker='o', markersize=4, linewidth=1.2)
plt.title('Primary Completion Rate')
plt.xlabel('Year $\longrightarrow$')
plt.ylabel('Primary completion rate(% of relevant age group)\
           $\longrightarrow$')
plt.legend(prop={'size': 8}, loc='center left',
           bbox_to_anchor=(1, .83), fontsize='large')
plt.show()


# Slicing the Dataframe
school = years.loc['School enrollment, primary and secondary (gross),\
                   gender parity index (GPI)',
                   '2005':'2010']


school = school.dropna(axis=0)

school = school.loc[['Belize', 'Colombia',
                     'Cyprus', 'Germany', 'Estonia'], '2005':'2010']

# Transpose the dataframe
school = school.T


# plotting the graph
matplotlib.rc('figure', figsize=(8, 6))
school.plot(linestyle=":", marker='o', markersize=4, linewidth=1.2)
plt.title('School enrollment, primary and secondary')
plt.xlabel('Year $\longrightarrow$')
plt.ylabel('School enrollment,primary & secondary (gross) $\longrightarrow$')
plt.legend(prop={'size': 8}, loc='center left',
           bbox_to_anchor=(1, .83), fontsize='large')
plt.show()
