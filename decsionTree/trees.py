from math import log
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

def chooseBestFeatureToSplit(dataset):
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








            




    




            
