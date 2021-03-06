import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv",delimiter=",")


def removeColumns(pandasArray,*columns):
    return pandasArray.drop(pandasArray.columns[[columns]], axis=1).values

def tagetAndtargetNames(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    target_names = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])

    for targetName in sorted(target_dict,key=target_dict.get):
        target_names.append(targetName)
    return np.asarray(target), target_names
target , targetNames = tagetAndtargetNames(my_data,1)
X = removeColumns(my_data,0,1)
y = target

neight = KNeighborsClassifier(n_neighbors=1)
neight.fit(X,y)
neight_7 = KNeighborsClassifier(n_neighbors=7)
neight_7.fit(X,y)

print (neight_7.predict(X[[30]]))
print (neight.predict(X[[30]]))

print (my_data)