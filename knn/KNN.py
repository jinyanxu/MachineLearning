from numpy import *
import os 
import operator

#a simple test of dataSet
def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#dataSet of datingTestSet2
def file2maxtrix(filename):
    fr = open(filename)
    arrayAllLines = fr.readlines()
    numberOfLines = len(arrayAllLines)
    group = zeros((numberOfLines,3))
    labels = []
    index = 0
    for line in arrayAllLines:
        line = line.strip()
        listFromLine = line.split('\t')
        group[index,:] = listFromLine[0:3]
        labels.append(int(listFromLine[-1]))
        index +=1
    return group,labels

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range (32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#dataSet normalization
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
    
#KNN
def classify0(inX, dataSet, labels, k): #输入，样本，样本标签，K
    dataSetSize = dataSet.shape[0]#获取数据集长度
    diffMat = tile(inX,(dataSetSize,1))-dataSet#计算待分类样本和每个样本集中的样本距离（差）
    sqDiffMat = diffMat**2#（平方）
    sqDistance = sqDiffMat.sum(axis=1)#（和）
    distance = sqDistance**0.5#(开根号)
    sortDistIndicies = distance.argsort()#排序索引
    classCount={}#dict类型
    for i in range(k):
        voteLaabel = labels[sortDistIndicies[i]]#距离最小的label(从小到大选k个)
        classCount[voteLaabel] = classCount.get(voteLaabel,0)+1#对应标签+1
    sortedClassCount = sorted(classCount.items(),key =operator.itemgetter(1),reverse = True)#按照出现频率排序
    return sortedClassCount[0][0]#返回频率最大的类别

#KNN datingClassTest
def datingClassTest():
    hoRatio = 0.15
    datingDataMat, datingLabels = file2maxtrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,],datingLabels[numTestVecs:m],3)
        if(classifierResult!=datingLabels[i]):
            errorCount +=1
            print("predict:%d,real:%d"%(classifierResult,datingLabels[i]))
    print("accuracy %f"%(1-errorCount/float(numTestVecs)))

#dating test
def classifyPerson():
    resultList = ["not at all","in small does","in large dose"]
    percentTags = float(input("玩视频游戏所占时间比:"))
    ffMiles = float(input("每年飞行里程数:"))
    iceCream = float(input("每周消耗冰淇淋公升数:"))
    datingDataMat,datingLabes = file2maxtrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTags,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabes,3)
    print("You will probably like this person：",resultList[classifierResult-1])

#handwritingClassTest
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')#文件名列表
    m = len(trainingFileList)#数据集长度
    trainingMat = zeros((m,1024))
    for i in range (m):
        filenameStr = trainingFileList[i]#每个文件的完整文件名0_0.txt、9_82.txt
        fileStr = filenameStr.split('.')[0]#文件名0_0、9_82
        classNumStr = filenameStr.split('_')[0]#数据类别
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' %filenameStr)#
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        filenameStr = testFileList[i]
        fileStr =filenameStr.split('.')[0]
        classNumStr = filenameStr.split('_')[0]
        vectorUnderTest = img2vector('digits/testDigits/%s' %filenameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        if(classifierResult != classNumStr):
            print("result:%d,real:%d"%(int(classifierResult),int(classNumStr)))
            errorCount +=1.0
    print('errorCount:%d:'%errorCount)
    print("totle accuracy:%f"%(1-errorCount/float(mTest)))
        





    
