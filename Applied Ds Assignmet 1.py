# Importing the python required packages
import numpy as np
import pandas as pd                    
import matplotlib.pyplot as plt

# Assignment 1 Question no.1 (Line Plot) Ans:-

# Reading the .csv(comma seprated value) file using Pandas pd.read_csv()
# read_csv() :- It is pandas function which allows to read the csv data 
                # files from directories.
museum_visit = pd.read_csv("Museum_Visitors.csv")

# Creating the DataFrame using Pd.DataFrame.
df_museum =pd.DataFrame(data=museum_visit)  

#Extracting the requierd no.of rows and columns by using iloc[] function  
# iloc[]:- Extract rows & columns by index number.            
df_museum = df_museum.iloc[36:48 , :5]

#Printing the data
 print(df_museum)

# Using Plt.figure() we can plot our chart.
plt.figure(figsize =(15,9))


# plt.plot() help to plot the line chart it takes two main arguments (x,y).
plt.plot(df_museum["Date"] , df_museum["Avila Adobe"],
         label="Avila Adobe Visits", linestyle = '-')
plt.plot(df_museum["Date"] , df_museum["Firehouse Museum"],
         label="Firehouse Museum Visits" , linestyle =':' )
plt.plot(df_museum["Date"] , df_museum["Chinese American Museum"],
        label="Chinese American Museum", linestyle ='--')
plt.plot(df_museum["Date"] , df_museum["America Tropical Interpretive Center"],
         label="America Tropical Center", linestyle = '-.')

# xlabel & ylabel() gives the label to X-axis & Y-axis respectively.
plt.xlabel("Months")
plt.ylabel("Number of Visitors (in thousands)")

# Using title() we can give title to our plot.
plt.title("Museum Visits in the Year 2017")

# legend():-describing the each part of the graph.
plt.legend()

# show():- With this function we can display our plot or graph.
 plt.show()





# Assignment 1 Question no.2 (Bar Plot) Ans:-

# Reading the .csv(comma seprated value) file using Pandas pd.read_csv()
d = pd.read_csv("Beijing Olympic 2022.csv")

# Creating the DataFrame using Pd.DataFrame.
olympic_data = pd.DataFrame(data=d) 
        
#Extracting the requierd no.of rows and columns by using loc[] function  
#loc[]:- Extract rows & columns by column name.      
df_total = olympic_data.loc[0:5,["NOC","Total"]]
print(df_total)


# passing color array for identify each country.
c=["lightblue","gray","green","pink","orange","cyan","brown","lightgreen"]

# Using Plt.figure() we can plot out chart
plt.figure(figsize=(20,6))

# plt.bar():- It helps to create the bar plot.
plt.bar(df_total["NOC"],df_total["Total"],color = c,width = 0.5)

# xlabel & ylabel() gives the label to X-axis & Y-axis respectively.
plt.xlabel("Name Of Countries")
plt.ylabel("Total Number of Medals Won")

# Using title() we can give title to our plot.
plt.title("Beijing Olympic 2022 Medals Won By Countries")

# show():- With this function we can display our plot or graph.
plt.show()




# Assignment 1 Question no.3 (Pie plot) Ans:-


# Reading the .csv(comma seprated value) file using Pandas pd.read_csv()
emp = pd.read_csv("Unemployment analysis.csv")

# Creating the DataFrame using Pd.DataFrame.
df_emp = pd.DataFrame(data=emp)

#Extracting the requierd no.of rows and columns by using loc[] function  
#loc[]:- Extract rows & columns by column name.
unemp = df_emp.loc[7:13 , ["Country Name","2000"]]

# creting list of country name & used in pass to labels.
name = unemp["Country Name"]

# Using Plt.figure() we can plot out chart.
plt.figure(figsize = (5,6))

# plt.pie():- It helps to create the pie chart.
plt.pie(unemp["2000"] , labels = name,autopct='%1.1f%%')

# Using title() we can give title to our plot.
plt.title("Unemployment rate in the year 2000")


#Second Pie chart of Year 2010


#Extracting the requierd no.of rows and columns by using loc[] function  
#loc[]:- Extract rows & columns by column name.
unemp1 = df_emp.loc[7:13 , ["Country Name" ,"2010"]]

# Using Plt.figure() we can plot out chart.
plt.figure(figsize = (5,6))

# plt.pie():- It helps to create the pie chart.
plt.pie(unemp1["2010"],labels = name ,autopct='%1.1f%%')

# Using title() we can give title to our plot.
plt.title("Unemployment rate in the year 2010")

# show():- With this function we can display our plot or graph.
plt.show()
