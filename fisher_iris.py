from csv import reader
import numpy as np
from sklearn.model_selection import KFold
from matplotlib.pyplot import *

"""打开数据集"""
filename ='iris.csv'
with open(filename,'rt',encoding='UTF-8')as raw_data:
    readers = reader(raw_data,delimiter=',')
    x = list(readers)
    data = np.array(x)
    #print(data)
    #print(data.shape)

data0=[]
def change_to_float(dataset):
    """将数组中的字符串转变成浮点数"""
    for i in dataset:
        i = list(map(float,i))
        data0.append(i)

def average(X_train,Y_train):
    """求类均值向量，其中X_train是训练集，Y_train是训练数据的对应类别"""
    m10 = np.zeros(4)
    n1 = 0
    m20 = np.zeros(4)
    n2 = 0

    for i in range(len(Y_train)):
        """分出不同的类别"""
        if Y_train[i] == 1:

            m10 = m10 + X_train[i]
            n1 += 1
        else:
            m20 = m20 + X_train[i]
            n2 += 1

    m1 = m10/n1
    m2 = m20/n2

    return m1,m2,n1,n2

def within_class(X_train,Y_train,m1,m2):
    """求总类内离散度矩阵"""
    S1 = np.zeros((4,4))
    S2 = np.zeros((4,4))

    for i in range(len(Y_train)):
        if Y_train[i] == 1:
            c1=np.array([X_train[i]-m1 ])
            S1+=c1.T@c1
        else:
            c2=np.array([(X_train[i]-m2) ])
            S2 += c2.T@c2

    Sw = S1+S2
    return Sw

def between_class(m1,m2,n1,n2):
    """求类间离散度矩阵"""
    m = (m1+m2)/2
    m1m=(m1-m).reshape(4,1)
    m2m=(m2-m).reshape(4,1)

    Sb =(n1*(np.dot(m1m,m1m.T))+n2*(np.dot(m2m,m2m.T)))/95
    return Sb

def draw_image(X_train,Y_train,w):
    for i in range(len(Y_train)):
        if Y_train[i] == 1:
            scatter(np.dot(X_train[i],w.T),'first')
        elif Y_train[i] == 2:
            scatter(np.dot(X_train[i],w.T),'second')
        else:
            scatter(np.dot(X_train[i],w.T),'third')
    show()

change_to_float(data)
data1=[] #类别数组
for i in range(len(data0)):
    """删除原数组的特征，变成类别数组"""
    data1.append(data0[i][4])
for i in range(len(data0)):
    """删除原数组的类别"""
    data0[i].pop()


"""k折交叉检验"""
X=np.array(data0)
Y=np.array(data1)
KF=KFold(n_splits=20,shuffle=True)
Accuracy = 0
for train_index,test_index in KF.split(X):
    X_train,X_test=X[train_index],X[test_index]
    Y_train,Y_test=Y[train_index],Y[test_index]
    #训练数据
    m1,m2,m3,n1,n2,n3 = average(X_train,Y_train)
    Sw = within_class(X_train,Y_train,m1,m2,m3)
    Sb = between_class(m1,m2,m3,n1,n2,n3)
    a,b = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
    w = b[0]
    y1 = np.dot(m1,w.T)
    y2 = np.dot(m2,w.T)
    y3 = np.dot(m3,w.T)
    g12 = (y1+y2)/2
    g23 = (y2+y3)/2
    print("\n")
    print("g12=",g12)
    print("g23=",g23)
    #测试数据
    N1 = 0
    N2 = 0
    N3 = 0
    if g12<g23:
        for i in range(len(Y_test)):
            if Y_test[i] == 1:
                if np.dot(X_test[i],w.T)<=g12:
                    N1=+1
            elif Y_test[i] == 2:
              print("accuracy=",accuracy)      if np.dot(X_test[i],w.T)<=g23 and np.dot(X_test[i],w.T)>=g12:
                    N2+=1
            else:
                if np.dot(X_test[i],w.T)>=g23:
                    N3+=1
    else:
        for i in range(len(Y_test)):
            if Y_test[i] == 1:
                if np.dot(X_test[i],w.T)>=g12:
                    N1=+1
            elif Y_test[i] == 2:
                if np.dot(X_test[i],w.T)>=g23 and np.dot(X_test[i],w.T)<=g12:
                    N2+=1
            else:
                if np.dot(X_test[i],w.T)<=g23:
                    N3+=1
    accuracy = (N1+N2)/5
    Accuracy += accuracy
    draw_image(X_train, Y_train, w)
print("\n")
print("Accuracy=",Accuracy/15)