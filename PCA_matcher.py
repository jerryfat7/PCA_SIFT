import numpy as np
from sklearn.decomposition import PCA
import boxes

import cv2
#for test
from PIL import Image
from detecter import detecter
from myDataSet import myDataSet
import evaluater
import time
import matplotlib.pyplot as plt
from mylogging import myLogger

class PCA_matcher():
    def __init__(self, n_components="mle"):
        self.n_components = n_components
        self.PCA_model = PCA(n_components = n_components)


    def train(self, pics, normalize=None):
        if normalize is None:
            normalize = PCA_matcher.Default_normalize

        #把每一个图片先归一化为2维矩阵（灰度图），然后拉成一条向量
        trainItemList = []
        for img in pics:
            trainItemList.append(normalize(img).reshape(-1))
        
        trainItemMat = np.stack(trainItemList)
        
        self.PCA_model.fit(trainItemMat)
        
        myLogger.info('所保留的n个成分各自的方差百分比:{}'.format(self.PCA_model.explained_variance_ratio_ ))
        myLogger.info('所保留的n个成分各自的方差值:{}'.format(self.PCA_model.explained_variance_  ))
        myLogger.info("方差总占比：{:.3f}".format(self.PCA_model.explained_variance_ratio_.sum()))
        return
    
    #计算相似度，采用距离作为相似度，norm为范数阶数
    def matchOneObj(self, base, target, norm=2):
        base_new = self.PCA_model.transform(base.reshape(1,-1))
        target_new = self.PCA_model.transform(target.reshape(1,-1))
        #print(base_new.shape)
        res = np.linalg.norm(target_new-base_new, norm)

        #print(res)
        return res

    def matchObjInFeature(self, baseFeature, targetFeature, norm=2):
        res = res = np.linalg.norm(baseFeature-targetFeature, norm)
        return res

    def transform(self, pic):
        assert len(pic.shape) == 2
        feature = self.PCA_model.transform(pic.reshape(1,-1))

        return feature


    def MatchObjInSecene(self, objs, sceneBboxes, scenesImgs, distanceThreshold, norm = 2, normalize = None):
        if normalize is None:
            normalize = PCA_matcher.Default_normalize
        
        objFeatures = []
        for objImg in objs:
            objFeatures.append(self.transform(objImg))

        res = []
        minDistancesForEachScene = -np.ones(len(scenesImgs))
        for i,sceneImg in enumerate(scenesImgs):
            #print("pictures：{}".format(i))
            bboxes = sceneBboxes[i]
            target_items = boxes.get_targetItems_by_boxes(sceneImg, np.array(bboxes).reshape(-1,4))
            #print(sceneImg.shape)
            #print(bboxes)
            if bboxes.size == 0:
                myLogger.info("scene id:{} no targets detected\n".format(i))
                res.append(-1)
                minDistancesForEachScene[i] = float("inf")
                continue

            #记录每个box与obj图片距离最小值
            distanceMinMatchForEachBox = np.zeros(bboxes.shape[0])

            for index in range(bboxes.shape[0]):
                target_item_feature = self.transform(normalize(target_items[index]))

                minDistance = float("inf")
                for objFeature in objFeatures:
                    distance = self.matchObjInFeature(objFeature, target_item_feature, norm)
                    if minDistance > distance:
                        minDistance = distance
                
                distanceMinMatchForEachBox[index] = minDistance

            minDistId = distanceMinMatchForEachBox.argmin()
            minDist = distanceMinMatchForEachBox[minDistId]
            
            minDistancesForEachScene[i] = minDist
            #print(minDist)
            if minDist > distanceThreshold:
                res.append(-1)
            else:
                res.append(minDistId)

        return np.array(res),minDistancesForEachScene
    
    #默认图片归一化函数，先转化为灰度图，然后做灰度拉伸和直方图均衡
    def Default_normalize(pic, width = 400, height = 400):
        #直方图均衡函数
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
        
        assert type(pic) == np.ndarray
        assert len(pic.shape) == 3
        
        #print(type(pic),pic.shape)
        assert type(pic) == np.ndarray and len(pic.shape) == 3
        newImg = cv2.resize(pic, (width,height),interpolation=cv2.INTER_CUBIC)
        newImg = cv2.cvtColor(newImg,cv2.COLOR_BGR2GRAY)
        #灰度拉伸
        minEl = newImg.min()
        maxEl = newImg.max()
        newImg = ((newImg - minEl)*(255/(maxEl - minEl))).astype(np.uint8)
        #直方图均衡
        newImg = clahe.apply(newImg)
               
        return newImg
    


if __name__ == "__main__":
    #加载数据集
    print("lodaing DataSet")
    start = time.time()
    #root = r"D:\学习\大四下\模式识别\大作业_瓶子检测\miniTest"
    root = r"D:\学习\大四下\模式识别\大作业_瓶子检测\Personalized_Segmentation"
    dataSet = myDataSet(root)
    #dataSet = myDataSet()
    trainMugs, trainScenes3, trainScenes5 = dataSet.getDataSet("train")
    
    
    #读取所有的特写照片
    objImgs = []
    for mugsFolder in trainMugs:
        objImgs += dataSet.ReadMugsPictures(mugsFolder)
    #SceneImgs = dataSet.ReadScenePictures(trainScenes3[1])
    print("训练集所用特写数量：{}".format(len(objImgs)))

    end = time.time()
    print("time usage:{:.2f}s".format(end-start))
    
    #目标检测预处理
    print("generationg Bboxes")
    start = time.time()

    mydet = detecter()
    #特写预处理,检测照片
    newObjImgs = []
    for ObjImg in objImgs:
        #目标检测，保留大于detectThreshold和数量小于等于maxCandidateItems的结果
        bboxes,scores = mydet.detect_sortedByScore(ObjImg, "cup")
        assert bboxes.shape[0] > 0
        cup_bbox = bboxes[0,:]

        newObjImg = boxes.get_targetItems_by_boxes(ObjImg, cup_bbox.reshape(-1,4))[0]
        
        newObjImgs.append(newObjImg)
        
    #训练PCA模型
    n_components = 35
    pcaModel = PCA_matcher(n_components)

    normalize = PCA_matcher.Default_normalize
    pcaModel.train(newObjImgs, normalize)
    
    end = time.time()
    print("time usage:{:.2f}s".format(end-start))
