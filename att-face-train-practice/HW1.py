import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
import cv2
from scipy.linalg import eigh
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import operator


data_folder = "att_faces/s"


def ReadFile(num, index):
    file_path = join(data_folder + str(num), str(index) + ".pgm")
    loaded_image = np.array(plt.imread(file_path))
    image_32x32 = cv2.resize(loaded_image, (32, 32),
                             interpolation=cv2.IMREAD_GRAYSCALE)
    image_1024 = image_32x32.reshape(-1)
    return image_1024


# Alternative to KNN
def getKNeighborsIndex(trainingSet, testInstance, k=1):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance.euclidean(testInstance, trainingSet[x])
        distances.append((trainingSet[x], x, dist))
    distances.sort(key=operator.itemgetter(2))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][1])
    return neighbors

# Alternative to accuracy_score method


def getAccuracy(testSet, predictions):
    correct = 0

    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


y_train = []
y_test = []

X_train = np.array([], dtype=int)
X_test = np.array([], dtype=int)

# reading data
for num in range(1, 41):
    trainList = np.array([1, 2, 3, 4, 5])
    testList = np.array([6, 7, 8, 9, 10])

    for index in trainList:
        image_1024 = ReadFile(num, index)
        X_train = np.concatenate((X_train, image_1024), axis=0)
        y_train.append(num)

    for index in testList:
        image_1024 = ReadFile(num, index)
        X_test = np.concatenate((X_test, image_1024), axis=0)
        y_test.append(num)

X_train = np.reshape(X_train, (200, 1024)).T
X_test = np.reshape(X_test, (200, 1024)).T


y_train = np.array(y_train)
y_test = np.array(y_test)
print("shape of X_train: ", np.shape(X_train))  # (1024,200)

# Standardization
stScaler = StandardScaler()
standardized_X_train = stScaler.fit_transform(X_train)
standardized_X_test = stScaler.transform(X_test)

# Calculating R_x
R_x = np.cov(standardized_X_train)

accuracies = []
number_of_features = range(5, 201, 5)

print("\nmodel accuracy based on number of chosen features in percent ( training on the 5 first image of each class ) :\n")

for number_of_feature in number_of_features:
    eigen_values, eigen_vectors = eigh(
        R_x, eigvals=(1024-number_of_feature, 1023))
    transform_matris = eigen_vectors

    # Apply learned transform matrix on train and test dataset
    reduced_X_train = np.matmul(transform_matris.T, standardized_X_train).T
    reduced_X_test = np.matmul(transform_matris.T, standardized_X_test).T

    # Using implemented KNN to predict test data
    y_pred = []
    for testInstance in reduced_X_test:
        neighbors = getKNeighborsIndex(reduced_X_train, testInstance)
        y_pred.append(y_test[neighbors[0]])

    # Calculating accuracy
    accuracies.append(accuracy_score(y_test, y_pred)*100)
    print("accuray on %d features is %.2f" %
          (number_of_feature, getAccuracy(y_test, y_pred)))

number_of_features_in_static_trainlist = number_of_features
accuracies_in_static_trainlist = accuracies


print(accuracies_in_static_trainlist)

fig = plt.figure(figsize=(20, 8))
plt.plot(number_of_features_in_static_trainlist,
         accuracies_in_static_trainlist)
plt.plot(number_of_features_in_static_trainlist,
         accuracies_in_static_trainlist, 'o')
plt.xlabel("Number of features")
plt.ylabel("Accuracy in %")
plt.xticks(number_of_features)
plt.grid()
plt.savefig("./static")


print("model accuracy based on number of chosen features in percent ( training on 5 random image of each class )  :\n")
# Changing number of selected features from 5 to 200
number_of_features = range(5, 201, 5)
accuracies = []
for number_of_feature in number_of_features:
    random_accuracies = []
    for epoch in range(20):
        y_train = []
        y_test = []

        X_train = np.array([], dtype=int)
        X_test = np.array([], dtype=int)

        # reading data
        for num in range(1, 41):
            trainList = np.random.choice(range(1, 11), 5, replace=False)
            testList = np.setdiff1d(range(1, 11), trainList)

            for index in trainList:
                image_1024 = ReadFile(num, index)
                X_train = np.concatenate((X_train, image_1024), axis=0)
                y_train.append(num)

            for index in testList:
                image_1024 = ReadFile(num, index)
                X_test = np.concatenate((X_test, image_1024), axis=0)
                y_test.append(num)

        X_train = np.reshape(X_train, (200, 1024)).T
        X_test = np.reshape(X_test, (200, 1024)).T

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Standardized
        stScaler = StandardScaler()
        standardized_X_train = stScaler.fit_transform(X_train)
        standardized_X_test = stScaler.transform(X_test)

        # Calculating R_x
        R_x = np.cov(standardized_X_train)  # (1024,1024)
        eigen_values, eigen_vectors = eigh(
            R_x, eigvals=(1024-number_of_feature, 1023))
        transform_matris = eigen_vectors
        reduced_X_train = np.matmul(transform_matris.T, standardized_X_train).T
        reduced_X_test = np.matmul(transform_matris.T, standardized_X_test).T

        # Using implemented KNN in order to predict test data
        y_pred = []
        for testInstance in reduced_X_test:
            neighbors = getKNeighborsIndex(reduced_X_train, testInstance)
            y_pred.append(y_test[neighbors[0]])
        random_accuracies.append(getAccuracy(y_test, y_pred))

    # Calculating mean of accuracies
    mean_of_acuracies = np.mean(random_accuracies)
    accuracies.append(mean_of_acuracies)
    print("average of accuray on %d features is %.2f" %
          (number_of_feature, mean_of_acuracies),)


number_of_features_in_random_trainlist = number_of_features
accuracies_in_random_trainlist = accuracies


print(accuracies_in_random_trainlist)

fig = plt.figure(figsize=(20, 8))
plt.plot(number_of_features_in_random_trainlist,
         accuracies_in_random_trainlist)
plt.plot(number_of_features_in_random_trainlist,
         accuracies_in_random_trainlist, 'o')
plt.xlabel("Number of features")
plt.ylabel("Accuracy in %")
plt.xticks(number_of_features)
plt.grid()
plt.savefig("/random")
