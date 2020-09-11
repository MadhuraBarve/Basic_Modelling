## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


## Setting up the path
path = 'D:/Madhura/Algorithms_Practice/Basic_Modelling/PCA/'

# Importing the data
df = pd.read_csv(path+'Wine.csv')

# check if all columns (except target column ["name" column in this case] are numerical)
# even if one of the features is of type that is "not" numerical e.g. string, obj etc .. then PCA cannot be executed
# df.info()

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

#  Getting predictor variables to X
X = df.drop(['name'],axis=1)

y= df['name']#class variable

## Standardization
x = StandardScaler().fit_transform(X)

## Finding variance covariance matrix
cov_x = np.cov(x.T)
# print(cov_x)

## Get Eigen values and eigen vectors from the variance covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_x)
print("Eigen values are: ",eig_vals)
print(eig_vecs.shape)
#eig_vals.sort(reverse = True)
#print("Sorted eigen values",eig_vals)

eig_pairs= [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0],reverse=True)
for i in eig_pairs:
	print(i[0])

### Finding contribution
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in eig_vals]
cum_var_exp = np.cumsum(var_exp)
print("cumulative variance explained: \n",cum_var_exp)

## Per expained by 1st PC
# print(eig_vals[0])
# print(eig_vals[0] / sum(eig_vals))
# print((eig_vals[0]+eig_vals[1]) / sum(eig_vals))

# Project data points to the eigen vector
#print(x.T.shape)	
# df_pc = x.dot(eig_vecs.T[:,:2])
# df_pc = pd.DataFrame(df_pc)
# df_pc.columns = ['PC1','PC2'] 
#df_pc.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13'] 
#print(df_pc.type)

matrix_w = np.hstack((eig_pairs[0][1].reshape(13,1),
	eig_pairs[1][1].reshape(13,1)))

print(matrix_w)

df_pc	= x.dot(matrix_w)
df_pc = pd.DataFrame(df_pc)
df_pc.columns = ['PC1','PC2'] 

data_viz_Df = pd.concat([df_pc, df[['name']]], axis = 1)
# print(data_viz_Df.head())

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
plt.show()
#plt.savefig(path+'Wine_scatter_PCA_Scratch.png')


		