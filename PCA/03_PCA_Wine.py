## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import warnings.filterwarnings('ignore')

## Setting up the path
path = 'D:/Madhura/Algorithms_Practice/Basic_Modelling/PCA/'

# Importing the data
df = pd.read_csv(path+'Wine.csv')

# check if all columns (except target column ["name" column in this case] are numerical)
# even if one of the features is of type that is "not" numerical e.g. string, obj etc .. then PCA cannot be executed
df.info()

# Exploratory Data Aanalysis(EDA)
# checking correlation inside the dataset
plt.figure(figsize = (8, 7))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
# plt.show()
plt.savefig(path+'Wine_Corrplot.png')

# pairplots another way of observing relationship between columns
# plt.figure(figsize=(8,7))
# sns.pairplot(df)
# plt.show()

# for purpose of viewing classification 
# we take 2 features with good correlation with Target Variable
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('flavanoids', fontsize = 15)
ax.set_ylabel('od280_od315', fontsize = 15)
ax.set_title('EDA View: Top 2 Features(Highest correlation with Target column [name]) from Original Dataframe', fontsize = 14)
targets = [1,2,3]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = df['name'] == target
    ax.scatter(df.loc[indicesToKeep, 'flavanoids']
               , df.loc[indicesToKeep, 'od280_od315']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
# plt.show()
plt.savefig(path+'Wine_scatter.png')

#  Getting predictor variables to X
X = df.drop(['name'],axis=1)

y= df['name']#class variable

#X.describe()

## Standardization
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(X)

# Let us view entire possible transformation for df under consideration
from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(x)
PC_df = pd.DataFrame(data = principalComponents)
print(PC_df.head())
print(pca.explained_variance_ratio_)

# for Data Vizualization we need only Top 2 features 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
PC_df = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2'])
print(PC_df.head())
df.head()

data_viz_Df = pd.concat([PC_df, df[['name']]], axis = 1)
print(data_viz_Df.head())


# code to plot classification on 2-D plane with PC1 and PC2
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1 (PC1)', fontsize = 15)
ax.set_ylabel('Principal Component 2 (PC2)', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1,2,3]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = data_viz_Df['name'] == target
    ax.scatter(data_viz_Df.loc[indicesToKeep, 'PC1']
               , data_viz_Df.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#plt.show()
plt.savefig(path+'Wine_scatter_PCA.png')

