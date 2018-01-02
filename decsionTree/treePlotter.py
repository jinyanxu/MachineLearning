import matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth",fc ="0.8")#创建字典
leftNode = dict(boxstyle = "round4",fc ="0.8")
arrow_args = dict(arrowstyle = "<-")
def plotNode(nodeNext,centerPt,patentPt,nodeType):
    createPlot.axl.annotate(nodeNext,xy=patentPt,xycoords="axes fraction",
                            xytext = centerPt,textcoords="axes fraction",
                            va ="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
def createPlot():
    fig = plt.figure(1,facecolor="white")
    fig.clf()
    createPlot.axl = plt.subplot(111,frameon =False)
    plotNode("-决策节点",(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode("-叶节点",(0.8,0.1),(0.3,0.8),leftNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.key()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.key():
        if type(secondDict[key])__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth  =0
    firstStr = myTree.key()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.key():
        if type(secondDict[key])__name__=='dict':
            thisDepth =1+getTreeDepth(secondDict[key])
        else:thisDepth=1
        if thisDepth>maxDepth:maxDepth=thisDepth
    return maxDepth





    
