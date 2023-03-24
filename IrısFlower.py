import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = 'pca_iris.data'
df = pd.read_csv(url, names=['sepal lenght', 'sepal width', 'petal lenght', 'target'])
df = df.reset_index()


features = ['sepal lenght', 'sepal width', 'petal lenght']
x = df[features]
y = df[['target']]

#Scale işlemi gerçekleştiriyoruz.
x= StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

print(principalDf)

final_dataframe = pd.concat([principalDf, y], axis=1)
print(final_dataframe)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

for target, col in zip(targets, colors):
    dftemp = final_dataframe[df.target == target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col)

#Veri setini koruyup korumadığımızı konrol ediyoruz.
exp = pca.explained_variance_ratio_
exp2 = pca.explained_variance_ratio_.sum()
print(exp, exp2)

plt.show()
