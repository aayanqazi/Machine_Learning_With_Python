import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
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

X_trainset,X_testset,y_trainset,y_testset = train_test_split(X,y,test_size=0.3,random_state=7)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh23 = KNeighborsClassifier(n_neighbors=23)
neigh90 = KNeighborsClassifier(n_neighbors=90)

neigh.fit(X_trainset,y_trainset)
neigh23.fit(X_trainset,y_trainset)
neigh90.fit(X_trainset,y_trainset)

pred = neigh.predict(X_testset)
pred23 = neigh23.predict(X_testset)
pred90 = neigh90.predict(X_testset)

print ("Neigh Accuracy",metrics.accuracy_score(y_testset,pred))
print ("Neight23 Accuracy", metrics.accuracy_score(y_testset,pred23))
print ("Neight90 Accuracy", metrics.accuracy_score(y_testset,pred90))



