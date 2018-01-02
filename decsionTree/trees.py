from math import log
import operator
#计算信息熵
def calcShannonEnt(dataset):
    numEntried = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntried
        shannonEnt -= prob*log(prob,2)#以2为底数求对数
    return shannonEnt

def createDataSet():
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataset,labels

#按照给定特征划分数据集(数据集、特征、特征值)        
def splitDataSet(dataset,axis,value):
    retDataset=[]
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec =featVec[:axis]#根据当前特征往前面取
            reducedFeatVec.extend(featVec[axis+1:])#往后面取加入reducedFeatVec
            retDataset.append(reducedFeatVec)#新的集合
    return retDataset

def chooseBestFeatureToSplit(dataset):#选出最佳属性进行分类
    numFeature = len(dataset[0])-1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain =0.0;bestFeature =-1
    for i in range(numFeature):
        featList = [example[i] for example in dataset]#所有样本的第i个特征
        uniqueVals = set(featList)#去除重复的特征，留下唯一的分类标签
        newEntropy = 0.0
        for value in uniqueVals:#对第i个特征分类计算信息熵
            subDataset = splitDataSet(dataset,i,value)
            prob = len(subDataset)/float(len(dataset))
            newEntropy += prob*calcShannonEnt(subDataset)
        infoGain = baseEntropy-newEntropy#信息增益
        if (infoGain>bestInfoGain):#选出最佳划分特征
            bestInfoGain = infoGain
            bestFeature =i
    return bestFeature

def majorityCnt(classList):
    classCount ={ }
    for vote in classCount:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reserve = True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]#类别列表
    if classList.count(classList[0]) ==len(classList):#类别完全相同，停止划分(第一个类别的个数==列表长度)
        return classList[0]
    if len(dataSet[0]) ==1:#遍历完所有特征返回出现次数最多的类别(dataset只剩下一个特征)
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)#选择最佳特征进行划分(特征序号)
    bestFeatLabel = labels[bestFeat]#(特征名)
    myTree = {bestFeatLabel:{}}#树的根节点(字典)
    del(labels[bestFeat])#从label中删除选择过的特征
    featValues = [example[bestFeat] for example in dataSet]#数据集中相应特征的值
    uniqueVals = set(featValues)#删除重复数据
    for value in uniqueVals:
        subLabels =labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree





            




    




            
