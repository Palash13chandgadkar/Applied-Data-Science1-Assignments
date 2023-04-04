import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('API_19_DS2_en_csv_v2_5361599.csv', skiprows=4)


def read_world_bank_data(filename, skip):

    data = pd.read_csv(filename, skiprows=skip)

    df_years = data.set_index(['Indicator Name', 'Country Name']).drop(
        ['Country Code', 'Indicator Code'], axis=1)

    df_countries = data.set_index(['Country Name']).drop(
        ['Country Code', 'Indicator Code'], axis=1).T

    return df_years, df_countries


df_data = read_world_bank_data('API_19_DS2_en_csv_v2_5361599.csv', 4)

years = df_data[0]
country = df_data[1]


renewable_eng = years.loc['Renewable energy consumption\
                          (% of total final energy consumption)',
                          '2005': '2010']


print(renewable_eng)

renewable_eng = renewable_eng.dropna(axis=0)

renewable_eng = renewable_eng.loc[['Australia', 'Belgium', 'Cameroon',
                                   'France', 'Georgia', 'India', 'Japan',
                                   'Malta', 'Libya', 'Mexico'], '2005':'2010']


plt.figure(figsize=(10, 25))
renewable_eng.plot.bar()
plt.title('Renewable energy consumption')


co2_emission = years.loc['CO2 emissions from liquid fuel consumption \
                         (% of total)',
                         '2005':'2010']


co2_emission.dropna(axis=0)

co2_emission = co2_emission.loc[['Australia', 'Belgium', 'Cameroon', 'France',
                                 'Georgia', 'India', 'Japan', 'Malta',
                                 'Libya', 'Mexico'], '2005':'2010']

print(co2_emission)


plt.figure(figsize=(10, 25))
co2_emission.plot.bar()
plt.title('co2 emission')
plt.show()
