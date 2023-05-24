from sklearn.svm import LinearSVC
import numpy as np

import os

# 根据线性SVM确认分类面，X1和X2分别为正样本和负样本的向量
def getClassifyPointSVM(X1, X2):
    X = np.hstack((X1.reshape(-1),X2.reshape(-1))).reshape(-1,1)
    y = np.hstack((np.zeros(X1.size),np.ones(X2.size)))
    linearsvc = LinearSVC(C = 1)
    
    linearsvc.fit(X,y)
    
    point = np.abs(linearsvc.intercept_[0] / linearsvc.coef_[0,0])
    
    return int(point)

def balancePositiveAndNegative(X1, X2, ratio = 1):
    assert X1.size > 0 and X2.size > 0
    assert ratio > 0

    np.random.shuffle(X1)
    np.random.shuffle(X2)
    if X1.size/X2.size >= ratio:
        new_X2 = X2
        new_X1 = X1[:int(X2.size*ratio)]
    else:
        new_X1 = X1
        new_X2 = X2[:int(X2.size*ratio)]

    return new_X1, new_X2


if __name__ == '__main__':
    #根据训练数据生成一个合理的分类面，用于判断杯子是否出现在场景中
    resultPath = r".\SIFT\train\sigma3_thresh7_ratio0.7"
    pdataPath = os.path.join(resultPath, "positive.npy")
    ndataPath = os.path.join(resultPath, "negative.npy")

    po = np.load(pdataPath).reshape(-1)
    # print(po.mean())
    # po.sort()
    # print(po[po.size//4])
    ne = np.load(ndataPath).reshape(-1)

    # print(ne.mean())
    # ne[-ne.size//4]


    new_po, new_ne = balancePositiveAndNegative(po, ne, 1)

    #print(new_po.shape, new_ne.shape)
    point = getClassifyPointSVM(ne,po)
    print(point)

