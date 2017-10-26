import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier

my_data = pandas.read_csv('https://ibm.box.com/shared/static/u8orgfc65zmoo3i0gpt9l27un4o0cuvn.csv')

#print (type(my_data))

# Display header of the data

#print (my_data.columns)

# Display Values of data without header

#print (my_data.values)

# Display dimension of the data

#print (my_data.shape)

# This function fixes my_data

def removeColumns(pandasArray,*columns):
    return pandasArray.drop(pandasArray.columns[[columns]], axis=1).values

new_data = removeColumns(my_data,0,1)
print (new_data)

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

target, target_names = tagetAndtargetNames(my_data,1)

X = new_data
y = target
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X,y)
print(neigh.predict(new_data[10]))

