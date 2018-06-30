from rootsift import RootSIFT
import cv2, glob, time, random
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

#read path image in dataset
def readData(path):
    files = glob.glob(path + "/*")
    imagePath = []
    for i, name in enumerate(files):
        imagePath.append(name)
    return imagePath

#get rootSift
def getRootSIFT(gray):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    # extract RootSIFT descriptors
    rs = RootSIFT()
    kp, des = rs.compute(gray, kp)
    return kp, des

#get descriptor of a image
def getDescriptors(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (0,0), fx=0.7, fy=0.7)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = getRootSIFT(gray)
    return des

# def createDesc(des_list):
#     # descriptors = des_list[0]
#     # for descriptor in des_list[1:]:
#     #     descriptors = np.vstack((descriptors, descriptor))
#     # return descriptors
#     descrip = np.vstack([des for des in des_list])
#     return descrip

#get list all descriptors of all image in dataset
def getAllDescriptors(path, desFile):
    imgpaths = readData(path)
    descsList = []
    for i in imgpaths:
        descsList.append(getDescriptors(i))
    descriptors = np.vstack(descsList)
    # descriptors = np.vstack([des for des in descsList])
    joblib.dump((imgpaths, descsList, descriptors), desFile, compress=3)
    return desFile

#training dataset
def trainning(desFile, numWord, trainFile):
    imgpaths, descsList, descriptors = joblib.load(desFile)

    model = MiniBatchKMeans(n_clusters=numWord, init_size=numWord*3, batch_size=1000,
            random_state=0).fit(descriptors)
    voc = model.cluster_centers_

    # Calculate the histogram of features
    im_features = np.zeros((len(imgpaths), numWord))

    for i in range(len(imgpaths)):
        words, distance = vq(descsList[i],voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(imgpaths)+1) / (1.0*nbr_occurences + 1)))

    # Perform L2 normalization
    im_features = im_features*idf
    im_features = preprocessing.normalize(im_features, norm='l2')
    joblib.dump((im_features, imgpaths, idf, numWord, voc), trainFile, compress=3)
    return trainFile

#get result of query
def getQueryResult(path):
    querys = []
    result = []
    pathquery = glob.glob(path)
    for q in pathquery:
        with open(q, 'r') as f:
            data = f.readline()
            temp = data.split()
            querys.append("ofquery/" + temp[0] +".jpg")
        with open(q.replace("query", "good"), 'r') as f:
            data = f.readlines()
            result.append(data)
    return querys, result

#compute Avarage Precision
def computeAP(Ytest, Ytrain):
    tu = 0
    mau = 0
    sum = 0
    for i in range(len(Ytest)):
        mau = mau + 1
        if Ytest[i] in Ytrain:
            tu = tu + 1
            sum = sum + tu/mau
    if tu:
        return sum/tu
    else:
        return 0

def drawMAP(pre,rec):
    map = sum(rec)/len(rec)
    plt.plot(pre,rec)
    plt.xlabel("Query")
    plt.ylabel("Average Precision")
    plt.title("MAP = {}%".format(round(map,0)*100))
    plt.show()
def main():
    st = time.time()
    if not glob.glob("desTest1.pkl"):
        print("Creating list descriptors...")
        desFile = getAllDescriptors("images", "desTest1.pkl")
    else:
        desFile = "desTest1.pkl"
    print("time create descriptors {}".format(time.time()-st))
    st = time.time()
    if not glob.glob("trainTest1.pkl"):
        print("training....")
        trainFile = trainning(desFile, 2000, "trainTest1.pkl")
    else:
        trainFile = "trainTest1.pkl"
    print("time training {}".format(time.time()-st))
    querys, result = getQueryResult("MAP" + "/*_query.txt")
    listDes = []
    for q in querys:
        listDes.append(getDescriptors(q))
    im_features, image_paths, idf, numWords, voc = joblib.load(trainFile)
    listResult = []
    for des in listDes:
        test_features = np.zeros((1, numWords))
        words, distance = vq(des,voc)
        for w in words:
            test_features[0][w] += 1
        # Perform Tf-Idf vectorization and L2 normalization
        test_features = test_features*idf
        test_features = preprocessing.normalize(test_features, norm='l2')
        score = np.dot(test_features, im_features.T)
        rank_ID = np.argsort(-score)
        temp = []
        for i in rank_ID[0][0:10]:
            img = cv2.imread(image_paths[i])
            temp.append(image_paths[i].split('/')[-1].replace(".jpg", "\n"))
        listResult.append(temp)

    s = 0
    listAP = []
    for i in range(len(listResult)):
        listAP.append(computeAP(listResult[i], result[i]))
    Q = np.arange(1, 56, 1)
    drawMAP(Q,listAP)


if __name__ == '__main__':
    main()
    # test()
