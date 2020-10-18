import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras import models
from keras.layers import Flatten
from keras import layers
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


symbol = ['M','R','L','P','V','C','I','G','A','Q','T','E','D','H','K','S','F','N','Y','W']

fastaTrain = open(r"C:\pythonProjects\thesis\Supp-S1.txt", "r")
fastaTest = open(r"C:\pythonProjects\thesis\Supp-A.txt", "r")
attributes = open(r"C:\pythonProjects\thesis\aaindex1.txt", "r")

def getDataFromTrainFile (file):
    seq=''
    accNumber=[]
    sequences=[]
    classes=[]
    dataList = [accNumber,sequences,classes]
    for line in file:
        line=line.strip()
        x=re.search('^>',line)
        y=re.search('^[A-Z]+$',line)
        z=re.search('^\(\d',line)
        if z!=None:
            seqClass=re.findall('^\(',line)
            dataList[2].append(line)
        elif x!=None:
            accNo=re.search('\_\S{6}',line)
            dataList[0].append(accNo.group()[1:])
            if seq!='':
                dataList[1].append(seq)
                seq=''
        elif y!=None:
            seq=seq+y.string
    dataList[1].append(seq)
    return dataList

    def getDataFromTestFile (file):
    seq=''
    accNumber=[]
    sequences=[]
    classes=[]
    dataList = [accNumber,sequences,classes]
    for line in file:
        line=line.strip()
        x=re.search('^>',line)
        y=re.search('[A-Z]{1,60}',line)
        z=re.search('^\(',line)
        if z!=None:
            seqClass=re.findall('^\(',line)
            dataList[2].append(line)
        elif x!=None:
            accNo=re.search('[^>]{6}',line)
            dataList[0].append(accNo.group())
            if seq!='':
                dataList[1].append(seq)
                seq=''
        elif y!=None:
            seq=seq+y.string
    dataList[1].append(seq)
    return dataList

trainDataset = getDataFromTrainFile(fastaTrain)
testDataset = getDataFromTestFile(fastaTest)

aminoAttr=[]
cnt=0
for line in attributes:
    line=line.strip()
    x=re.search('^-?\d+\.\d*',line)
    if x!=None:
        cnt+=1
        attribs=re.findall('-?\d+\.\d*',line)
        aminoAttr.append(attribs)

floatList=[]
tempList=[]
for i,item in enumerate(aminoAttr):
   for j in item:
      tempList.append(float(j))
   if i%2!=0:
      floatList.append(tempList)
      tempList=[]

myDf = pd.DataFrame(floatList).transpose()
rows,cols = myDf.shape
for i in range(cols):
    myDf[i].fillna(myDf[i].median(),inplace=True)

pca = PCA(n_components=18)
reducedData = pca.fit_transform(myDf)
reducedData = preprocessing.scale(reducedData)

fastaTest.close
fastaTrain.close
attributes.close        

a=np.zeros(626)
b=np.ones(299)*1
c=np.ones(42)*2
d=np.ones(73)*3
e=np.ones(2437)*4
f=np.ones(403)*5
g=np.ones(172)*6
h=np.ones(1450)*7
classes = np.concatenate([a,b,c,d,e,f,g,h],axis=0)

trainDf = pd.DataFrame(zip(trainDataset[0],trainDataset[1],classes),columns=['accNo','Sequence','Type'])

a=np.zeros(610)
b=np.ones(312)*1
c=np.ones(24)*2
d=np.ones(44)*3
e=np.ones(1316)*4
f=np.ones(151)*5
g=np.ones(182)*6
h=np.ones(610)*7
classes = np.concatenate([a,b,c,d,e,f,g,h],axis=0)

testDf = pd.DataFrame(zip(testDataset[0],testDataset[1],classes),columns=['accNo','Sequence','Type'])

trainDf['Type'] = trainDf['Type'].astype(np.int8,copy=False)
testDf['Type'] = testDf['Type'].astype(np.int8,copy=False)

condition1 = trainDf['Length']<=1500
train1500 = trainDf[condition1]
train1500 = train1500.drop(['Length','accNo'],1)
train1500 = train1500.reset_index(drop=True)

condition2 = train1500['Sequence'].apply(lambda x: ('X' not in x) and ('B' not in x) and ('Z' not in x) and ('U' not in x))
train1500 = train1500[condition2]
trainRows,trainCols=train1500.shape

condition1 = testDf['Length']<=1500
test1500 = testDf[condition1]
test1500 = test1500.drop(['Length','accNo'],1)
test1500 = test1500.reset_index(drop=True)

condition2 = test1500['Sequence'].apply(lambda x: ('X' not in x) and ('B' not in x) and ('Z' not in x))
test1500 = test1500[condition2]
testRows,testCols=test1500.shape

cnt=0
oneHotDic={}
for i,aa in enumerate(train1500.iloc[0,0]):
    if aa not in oneHotDic:
        oneHotDic[aa]=cnt
        cnt+=1

physicoDictionary = {}
for i in range(20):
    physicoDictionary[symbol[i]]=np.around(reducedData[i,:],2)

np.random.seed(5)

trainArray = train1500.to_numpy()
np.random.shuffle(trainArray)
trainSequences = trainArray[:,0]
trainClasses = trainArray[:,1]
trainClasses=trainClasses.astype(np.int8)

testArray = test1500.to_numpy()
np.random.shuffle(testArray)
testSequences = testArray[:,0]
testClasses = testArray[:,1]
testClasses=testClasses.astype(np.int8)

flag=True
for seq in trainSequences:
    trainRow=[0]*30000
    for i,aa in enumerate(seq):
        trainRow[oneHotDic[aa]+i*20] = 1
    if flag:
        trainForML = trainRow
        flag=False
    else:
        trainForML = np.vstack((trainForML,trainRow))  

flag=True
for seq in testSequences:
    testRow=[0]*30000
    for i,aa in enumerate(seq):
        testRow[oneHotDic[aa]+i*20] = 1
    if flag:
        testForML = testRow
        flag=False
    else:
        testForML = np.vstack((testForML,testRow))  

trainSet = np.zeros([trainSequences.shape[0],1500,len(oneHotDic)])
for i,seq in enumerate(trainSequences):
    for j,aa in enumerate(seq):
        trainSet[i,j,oneHotDic[aa]]=1

testSet = np.zeros([testSequences.shape[0],1500,len(oneHotDic)])
for i,seq in enumerate(testSequences):
    for j,aa in enumerate(seq):
        testSet[i,j,oneHotDic[aa]]=1

trainSampleweight = class_weight.compute_sample_weight('balanced',trainClasses)

clfDT = tree.DecisionTreeClassifier(max_depth=10,min_samples_split=2,min_samples_leaf=1, class_weight='balanced')
clfNB = GaussianNB()
clfNN = KNeighborsClassifier(n_neighbors=4,weights='distance',metric='minkowski')
clfSVM= svm.LinearSVC(class_weight='balanced')

clfDT.fit(trainForML, trainClasses, sample_weight=trainSampleweight)
clfNB.fit(trainForML, trainClasses, sample_weight=trainSampleweight)
clfNN.fit(trainForML, trainClasses, sample_weight=trainSampleweight)
clfSVM.fit(trainForML, trainClasses, sample_weight=trainSampleweight)

y_test_pred_DT=clfDT.predict(testForML)
y_test_pred_NB=clfNB.predict(testForML)
y_test_pred_NN=clfNN.predict(testForML)
y_test_pred_SVM=clfSVM.predict(testForML)

precisionDT=precision_recall_fscore_support(testClasses, y_test_pred_DT, average='macro')[0]
precisionNB=precision_recall_fscore_support(testClasses, y_test_pred_NB, average='macro')[0]
precisionNN=precision_recall_fscore_support(testClasses, y_test_pred_NN, average='macro')[0]
precisionSVM=precision_recall_fscore_support(testClasses, y_test_pred_SVM, average='macro')[0]

trainClasses = to_categorical(trainClasses,num_classes=8,dtype='int8')
testClasses = to_categorical(testClasses,num_classes=8,dtype='int8')

trainSeq = trainSet[:3000,:1500,:20]
valSeq = trainSet[3000:,:1500,:20]
trainClass = trainClasses[:3000,:8]
valClass = trainClasses[3000:,:8]

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(1500,20)))
model.add(layers.Dense(16, activation='relu'))
model.add(Flatten())
model.add(layers.Dense(8, activation='softmax'))

model = models.Sequential()
model.add(layers.Conv1D(128, (15), activation='relu',
input_shape=(1500,20)))
model.add(layers.MaxPooling1D((2)))
model.add(layers.Conv1D(64, (5), activation='relu'))
model.add(layers.MaxPooling1D((2)))
model.add(layers.Conv1D(32, (2), activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(8, activation='softmax'))

model = models.Sequential()
model.add(layers.Conv1D(128, (15),strides=10, activation='relu',
input_shape=(1500,20)))
model.add(layers.MaxPooling1D((2)))
model.add(layers.LSTM(64))
model.add(layers.Flatten())
model.add(layers.Dense(8, activation='softmax'))

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(1500,20)))
for rate in (1, 2, 4, 8):
    model.add(layers.Conv1D(64, (15),
    padding="causal",activation="relu", dilation_rate=rate))
model.add(layers.Conv1D(64,(2)))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(8, activation='softmax'))

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

history = model.fit(trainSeq,trainClass,
epochs=20,
batch_size=64,
sample_weight=trainSampleWeight,
validation_data=(valSeq, valClass))

model.evaluate(testSet,testClasses,batch_size=64,sample_weight=testSampleWeight)
