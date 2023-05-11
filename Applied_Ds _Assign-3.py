# Importing packages

import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cluster_tools as ct
import pandas as pd

# Extracting data
data_edu = pd.read_csv("education.csv", skiprows=4)

indicator = data_edu[data_edu['Indicator Name'].isin(
    ['School enrollment, primary (% gross)'])]

school_ind = indicator.loc[:, [
    'Country Name', 'Indicator Name', '2001', '2018']]

school_ind = school_ind.dropna()

school_ind.info()
print(school_ind)


scaler = StandardScaler()
school_scale_data = scaler.fit_transform(school_ind[['2001', '2018']])
school_scale_data = pd.DataFrame(school_scale_data, columns=['2001', '2018'])
school_scale_data.describe().round(2)

a = []


for k in range(2, 11):
    kmean = KMeans(n_clusters=k, n_init=25, random_state=1234)
    kmean.fit(school_scale_data)
    a.append(kmean.inertia_)

a_series = pd.Series(a, index=range(2, 11))

# plt.figure(figsize=(8, 6))
# ax = sns.lineplot(y=wcss_series, x=wcss_series.index)
# ax = sns.scatterplot(y=wcss_series, x=wcss_series.index, s=150)
# ax = ax.set(xlabel='Number of Clusters (k)',
#             ylabel='Within Cluster Sum of Squares (WCSS)')


kmean = KMeans(n_clusters=5, n_init=25, random_state=1234)
kmean.fit(school_scale_data)
centers = kmean.cluster_centers_

simpleClusterInfo = pd.Series(kmean.labels_).value_counts().sort_index()
print(simpleClusterInfo)


centers = kmean.cluster_centers_
xcen = centers[:, 0]
ycen = centers[:, 1]


plt.figure(figsize=(10, 8))
ax = sns.scatterplot(data=school_scale_data, x='2001', y='2018',
                     hue=kmean.labels_, palette='deep',
                     alpha=0.8, s=80, legend=True)
plt.title("Access to Education Around The World ")
plt.scatter(xcen, ycen, 40, "r", marker="d")


# Curve-fit
# Assuming you have x and y data arrays
x = school_scale_data['2001']
y = school_scale_data['2018']

# Define the function to fit the curve


def my_func(x, a, b, c):
    return a * np.exp(-b * x) + c


# Perform the curve fitting
popt, pcov = curve_fit(my_func, x, y)

# Retrieve the fitted parameters
a_fit, b_fit, c_fit = popt

# Generate the curve based on the fitted parameters
x_curve = np.linspace(min(x), max(x), 100)
y_curve = my_func(x_curve, a_fit, b_fit, c_fit)

# Plot the original data and the fitted curve
plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x_curve, y_curve, 'r-', label='Fitted Curve')
plt.xlabel('2001')
plt.ylabel('2018')
plt.legend()
plt.title("Curve Fit")
plt.show()


# side-by-side plot
school_ind['Cluster'] = kmean.labels_.tolist()


ct1 = school_ind[school_ind['Cluster'].isin([0])]
ct2 = school_ind[school_ind['Cluster'].isin([1])]
ct3 = school_ind[school_ind['Cluster'].isin([2])]
ct4 = school_ind[school_ind['Cluster'].isin([3])]
ct5 = school_ind[school_ind['Cluster'].isin([4])]

ct1 = ct1.set_index('Country Name')
ct2 = ct2.set_index('Country Name')
ct3 = ct3.set_index('Country Name')
ct4 = ct4.set_index('Country Name')
ct5 = ct5.set_index('Country Name')


m_data = ct1.loc['Morocco', '2001']
a_data = ct2.loc['Armenia', '2001']
f_data = ct3.loc['Afghanistan', '2001']
n_data = ct4.loc['Niger', '2001']
t_data = ct5.loc['Togo', '2001']

m_data18 = ct1.loc['Morocco', '2018']
a_data18 = ct2.loc['Armenia', '2018']
f_data18 = ct3.loc['Afghanistan', '2018']
n_data18 = ct4.loc['Niger', '2018']
t_data18 = ct5.loc['Togo', '2018']

print(m_data18)


data_2001 = [m_data, a_data, f_data, n_data, t_data]


data_2018 = [m_data18, a_data18, f_data18, n_data18, t_data18]


countries = ['Mozambique', 'Niger', 'Armenia', 'Burundi', 'Afganistan']
x = np.arange(len(countries))
width = 0.25

data_2001 = [m_data, a_data, f_data, n_data, t_data]
data_2018 = [m_data18, a_data18, f_data18, n_data18, t_data18]


fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, data_2001, width, label='2001')
rects2 = ax.bar(x + width/2, data_2018, width, label='2018')

ax.set_xlabel('Country')
ax.set_ylabel('Value')
ax.set_title('Comparison of Data between 2001 and 2018')
ax.set_xticks(x)
ax.set_xticklabels(countries)
ax.legend()

plt.show()
