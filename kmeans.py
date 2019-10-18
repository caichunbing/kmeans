#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : pycharm
#   File name   : kmeans.py
#   Author      : caichunbing
#   Created date: 2019-10-18 
#   Description :kmeans聚类算法及可视化
#
#================================================================


import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        curLine=[float(x) for x in curLine]
        # fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(curLine)
    return np.mat(dataMat)


def distEclud(vecA, vecB):
    dist=np.sqrt(np.sum(np.power(vecA - vecB, 2)))
    return dist  # la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])

                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment


def show(w,h,centroid_w,centroid_h):
    fig = plt.figure()
    fig.suptitle("kmeans")

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(w, h, s=10, color='b')
    ax1.scatter(centroid_w,centroid_h,s=10,color='r')

    plt.show()

filename="./testSet.txt"

if __name__ == '__main__':
    dataSet=loadDataSet(filename)
    centoid,cluster=kMeans(dataSet, 4,distEclud,randCent)

    w = dataSet[:, 0].tolist()
    h = dataSet[:, 1].tolist()
    centoid_w=centoid[:,0].tolist()
    centoid_h=centoid[:,1].tolist()


    show(w,h,centoid_w,centoid_h)