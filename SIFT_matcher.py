import cv2
#import matplotlib.pyplot as plt
# import PIL
import numpy as np

import boxes
#import torch

#for test
from PIL import Image
from detecter import detecter
from myDataSet import myDataSet
import evaluater
import time
import traceback
from mylogging import myLogger

# 输入是一组物品的特写以及场景的截取图
# objs = [array1, array2,.., arrayn]
# scenes = [[target_obj1,...,target_objn],...]
# points_threshold用于确定图片与图片间是否匹配，关键点数量大于等于points_threshold可认为匹配
# 一个场景里的杯子需要有多个特写图片与之匹配，才可认为该场景中识别到的物体就是该物体
class SIFT_matcher():
    def __init__(self, sift = None):
        if sift is None:
            self.sift = cv2.xfeatures2d.SIFT_create()
        else:
            self.sift = sift

    #return keypoints,descriptors
    def detectAndCompute(self,img:np.ndarray):
        return self.sift.detectAndCompute(img, None)
    
    # 特征向量判定距离默认采用一范数，可选：NORM_L1 ,NORM_L2 ,NORM_L2SQR ,NORM_INF, NORM_HAMMING ,NORM_HAMMING2  
    #imgMode:{"Gray","RGB","HSV"}
    def MatchObjInSeceneUsingMaxMatches(self, objs, sceneBboxes, scenesImgs,pointsThreshold,picturesThreshold, 
                                        ratio = 0.7, normType = cv2.NORM_L1,
                                        MatchesNumPrior = False, usingFlann=True, imgMode="Gray"):
        """
        input:objs,一个list，每个元素为同一个杯子的多角度图片
        sceneBboxes：一个场景里杯子的bboxes，是一个list，其中每一个元素为一个numpy/Tensor
        (x1,y1,x2,y2)格式的bbox
        scnensImgs：一个list，每个list中是一个numpy/Tensor格式的场景图片
        
        """
        
        #匹配器
        #flann算法匹配
        if usingFlann:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=40)
            matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            matcher = cv2.BFMatcher(normType, crossCheck=True)
        # 匹配结果储存,若obj出现在该场景图中则对应位置元素为index，即scene中target_item的索引
        # 不存在则对应位置为-1
        res = []

        #先计算所有的特写图片的特征向量
        objInfos = []
        for obj in objs:
            keypoints = dict()
            descriptors = dict()
            #print(obj.shape)
            if imgMode == "RGB":
                keypoints["R"], descriptors["R"] = self.detectAndCompute(obj[:,:,0])
                keypoints["G"], descriptors["G"] = self.detectAndCompute(obj[:,:,1])
                keypoints["B"], descriptors["B"] = self.detectAndCompute(obj[:,:,2])
            elif imgMode == "Gray":
                t_obj = cv2.cvtColor(obj,cv2.COLOR_BGR2GRAY)
                keypoints["Gray"],descriptors["Gray"] = self.detectAndCompute(t_obj)
            elif imgMode == "HSV":
                temp_obj = cv2.cvtColor(obj,cv2.COLOR_BGR2HSV)
                h_obj, s, v = cv2.split(temp_obj)
                #仅使用H通道
                keypoints["H"],descriptors["H"] = self.detectAndCompute(h_obj)
                print(h_obj)
                print(descriptors["H"].shape)
            else:
                raise NotImplementedError
                
            objInfos.append([keypoints, descriptors])

        MaxMatchesEachScene = np.zeros(len(scenesImgs))
        for i,sceneImg in enumerate(scenesImgs):
            
            bboxes = sceneBboxes[i]
            target_items = boxes.get_targetItems_by_boxes(sceneImg, np.array(bboxes).reshape(-1,4))
            #print(sceneImg.shape)
            #print(bboxes)
            if bboxes.size == 0:
                myLogger.info("scene id:{} no targets detected\n".format(i))
                res.append(-1)
                continue
            #记录每个box与obj图片的matches匹配数量
            picturesForEachBox = np.zeros(bboxes.shape[0])
            #记录每个box与obj图片向量匹配的最大数量
            scoresMatchForEachBox = np.zeros(bboxes.shape[0])
            
            #依次匹配obj的特征向量与scene中检测的target的特征向量
            for index in range(bboxes.shape[0]):
                target_item = target_items[index]
                #print(bboxes[index])
                #print(target_item.shape)
                keypoints = dict()
                descriptors = dict()
                #for key in objInfos[0][0].keys():
                if imgMode == "RGB":
                    keypoints["R"], descriptors["R"] = self.detectAndCompute(target_item[:,:,0])
                    keypoints["G"], descriptors["G"] = self.detectAndCompute(target_item[:,:,1])
                    keypoints["B"], descriptors["B"] = self.detectAndCompute(target_item[:,:,2])
                elif imgMode == "Gray":
                    t_target_item = cv2.cvtColor(target_item,cv2.COLOR_BGR2GRAY)
                    keypoints["Gray"],descriptors["Gray"] = self.detectAndCompute(t_target_item)
                elif imgMode == "HSV":
                    temp_target_item = cv2.cvtColor(target_item,cv2.COLOR_BGR2HSV)
                    h_target, s, v = cv2.split(temp_target_item)
                #仅使用H通道
                    keypoints["H"],descriptors["H"] = self.detectAndCompute(h_target)
                    print(descriptors["H"].shape)
                    
                
                pic_match_counter = 0
                maxMatches = -1
                #每张特写图片
                for objInfo in objInfos:
                    matchesCount = 0
                    for channel,key in enumerate(objInfo[0].keys()):
                        successFlag = True
                        try:
                            matches = matcher.knnMatch(descriptors[key], objInfo[1][key],k=2)
                        except Exception as e:
                            myLogger.error(e)
                            myLogger.error("knn调用出错")
                            traceback.print_exc()
                            successFlag = False


                        goodMatches = []
                        if successFlag:
                            for m,n in matches:
                                #sift距离匹配的阈值
                                if m.distance < ratio * n.distance:
                                    goodMatches.append(m)
                        matchesCount += len(goodMatches)
                    #增加匹配数达到阈值的图片数量
                    if matchesCount >= pointsThreshold:
                        pic_match_counter += 1
                    
                    #更新最大匹配数
                    if matchesCount > maxMatches:
                        maxMatches = matchesCount
                    
                picturesForEachBox[index] = pic_match_counter
                scoresMatchForEachBox[index] = maxMatches

            MaxMatchesEachScene[i] = scoresMatchForEachBox.max()  
            #MatchesNumPrior代表优先使用匹配数进行分类，即考虑大于pictureThreshold但是picturesForEachBox最大的一类
            #若为False则考虑大于
            if not MatchesNumPrior:
                # 选取匹配数量最多且大于pictureThreshold的认为是出现在了场景中，
                # 记录该在sceneBboxes[i]中的索引
                BestMatchBboxId = picturesForEachBox.argmax()
                if picturesForEachBox[BestMatchBboxId] >= picturesThreshold:
                    res.append(BestMatchBboxId)
                else:
                    res.append(-1)
            else:
                # 选取匹配数量最多且大于pictureThreshold的认为是出现在了场景中，
                # 记录该在sceneBboxes[i]中的索引
                
                BestMatchBboxId = scoresMatchForEachBox.argmax()
                if picturesForEachBox[BestMatchBboxId] >= picturesThreshold:
                    res.append(BestMatchBboxId)
                else:
                    res.append(-1)
                
        return np.array(res),MaxMatchesEachScene
    
    # 特征向量判定距离默认采用一范数，可选：NORM_L1 ,NORM_L2 ,NORM_L2SQR ,NORM_INF, NORM_HAMMING ,NORM_HAMMING2  
    #imgMode:{"Gray","RGB","HSV"}
    def MatchObjInSeceneUsingMaxMatches_v1(self, objs, sceneBboxes, scenesImgs,pointsThreshold,picturesThreshold, 
                                        ratio = 0.7, normType = cv2.NORM_L1,
                                        MatchesNumPrior = False, usingFlann=True, imgMode="Gray"):
        """
        input:objs,一个list，每个元素为同一个杯子的多角度图片
        sceneBboxes：一个场景里杯子的bboxes，是一个list，其中每一个元素为一个numpy/Tensor
        (x1,y1,x2,y2)格式的bbox
        scnensImgs：一个list，每个list中是一个numpy/Tensor格式的场景图片
        
        """
        
        #匹配器
        #flann算法匹配
        if usingFlann:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            matcher = cv2.BFMatcher(normType, crossCheck=True)
        # 匹配结果储存,若obj出现在该场景图中则对应位置元素为index，即scene中target_item的索引
        # 不存在则对应位置为-1
        res = []

        #先计算所有的特写图片的特征向量
        objInfos = []
        for obj in objs:
            keypoints = dict()
            descriptors = dict()
            #print(obj.shape)
            if imgMode == "RGB":
                keypoints["R"], descriptors["R"] = self.detectAndCompute(obj[:,:,0])
                keypoints["G"], descriptors["G"] = self.detectAndCompute(obj[:,:,1])
                keypoints["B"], descriptors["B"] = self.detectAndCompute(obj[:,:,2])
            elif imgMode == "Gray":
                t_obj = cv2.cvtColor(obj,cv2.COLOR_BGR2GRAY)
                keypoints["Gray"],descriptors["Gray"] = self.detectAndCompute(t_obj)
            elif imgMode == "HSV":
                temp_obj = cv2.cvtColor(obj,cv2.COLOR_BGR2HSV)
                h_obj, s, v = cv2.split(temp_obj)
                #仅使用H通道
                keypoints["H"],descriptors["H"] = self.detectAndCompute(h_obj)
                print(h_obj)
                print(descriptors["H"].shape)
            else:
                raise NotImplementedError
                
            objInfos.append([keypoints, descriptors])

        MaxMatchesEachScene = np.zeros(len(scenesImgs))
        for i,sceneImg in enumerate(scenesImgs):
            
            bboxes = sceneBboxes[i]
            #target_items = boxes.get_targetItems_by_boxes(sceneImg, np.array(bboxes).reshape(-1,4))
            #print(sceneImg.shape)
            #print(bboxes)
            if bboxes.size == 0:
                print("scene id:{} no targets detected\n".format(i))
                res.append(-1)
                continue
            
            
            SceneKeypoints = dict()
            SceneDescriptors = dict()
            #for key in objInfos[0][0].keys():
            if imgMode == "RGB":
                SceneKeypoints["R"], SceneDescriptors["R"] = self.detectAndCompute(sceneImg[:,:,0])
                SceneKeypoints["G"], SceneDescriptors["G"] = self.detectAndCompute(sceneImg[:,:,1])
                SceneKeypoints["B"], SceneDescriptors["B"] = self.detectAndCompute(sceneImg[:,:,2])
            elif imgMode == "Gray":
                t_sceneImg = cv2.cvtColor(sceneImg,cv2.COLOR_BGR2GRAY)
                SceneKeypoints["Gray"],SceneDescriptors["Gray"] = self.detectAndCompute(t_sceneImg)
            elif imgMode == "HSV":
                temp_sceneImg = cv2.cvtColor(sceneImg,cv2.COLOR_BGR2HSV)
                h_scene, s, v = cv2.split(temp_sceneImg)
            #仅使用H通道
                SceneKeypoints["H"],SceneDescriptors["H"] = self.detectAndCompute(h_scene)
                print(descriptors["H"].shape)
            
            
            
            
            #记录每个box与obj图片的matches匹配数量
            picturesForEachBox = np.zeros(bboxes.shape[0])
            #记录每个box与obj图片向量匹配的最大数量
            scoresMatchForEachBox = np.zeros(bboxes.shape[0])
            #计算场景中所有特征向量与每张特写的匹配情况，筛选匹配结果，并根据特征点位置确定每个
            #特征向量的匹配是属于哪个Bbox的，获得匹配结果最多的特征框
            for objInfo in objInfos:

                #直接match
                matchesDict = dict()

                
                #每个Bbox对该特写图片的匹配向量数
                matchesCount = np.zeros(bboxes.shape[0])
                for channel,key in enumerate(objInfo[0].keys()):
                    successFlag = True
                    try:
                        matches = matcher.knnMatch(SceneDescriptors[key], objInfo[1][key],k=2)
                    except Exception as e:
                        myLogger.error(e)
                        myLogger.error("knn调用出错")
                        traceback.print_exc()
                        successFlag = False
                    
                    
                    goodMatches = [[] for j in range(bboxes.shape[0])]
                    if successFlag:
                        for m,n in matches:
                            #sift距离匹配的阈值
                            if m.distance < ratio * n.distance:
                                for j in range(bboxes.shape[0]):
                                    keyPoint = SceneKeypoints[key][m.queryIdx]
                                    if boxes.pointInBox(keyPoint.pt[0],keyPoint.pt[1], bboxes[j,:]):
                                        goodMatches[j].append(m)
                    
                    for j in range(bboxes.shape[0]):
                        matchesCount[j] += len(goodMatches[j])
                    
                #增加匹配数达到阈值的图片数量
                bboxId = np.where(matchesCount >= pointsThreshold)[0]
                picturesForEachBox[bboxId] += 1


                #更新最大匹配数
                for j in range(bboxes.shape[0]):
                    if matchesCount[j] > scoresMatchForEachBox[j]:
                        scoresMatchForEachBox[j] = matchesCount[j]

            MaxMatchesEachScene[i] = scoresMatchForEachBox.max()
            #MatchesNumPrior代表优先使用匹配数进行分类，即考虑大于pictureThreshold但是picturesForEachBox最大的一类
            #若为False则考虑大于
            if not MatchesNumPrior:
                # 选取匹配数量最多且大于pictureThreshold的认为是出现在了场景中，
                # 记录该在sceneBboxes[i]中的索引
                BestMatchBboxId = picturesForEachBox.argmax()
                if picturesForEachBox[BestMatchBboxId] >= picturesThreshold:
                    res.append(BestMatchBboxId)
                else:
                    res.append(-1)
            else:
                # 选取匹配数量最多且大于pictureThreshold的认为是出现在了场景中，
                # 记录该在sceneBboxes[i]中的索引

                BestMatchBboxId = scoresMatchForEachBox.argmax()
                if picturesForEachBox[BestMatchBboxId] >= picturesThreshold:
                    res.append(BestMatchBboxId)
                else:
                    res.append(-1)
                
        return np.array(res),MaxMatchesEachScene
    # 特征向量判定距离默认采用一范数，可选：NORM_L1 ,NORM_L2 ,NORM_L2SQR ,NORM_INF, NORM_HAMMING ,NORM_HAMMING2  
    #imgMode:{"Gray","RGB","HSV"}
    def MatchObjInSeceneUsingMaxPoints(self, objs, sceneBboxes, scenesImgs, ratio = 0.7, normType = cv2.NORM_L1,
                         detectThreshold = 0.5,usingFlann=True, imgMode="Gray"):
        """
        input:objs,一个list，每个元素为同一个杯子的多角度图片
        sceneBboxes：一个场景里杯子的bboxes，是一个list，其中每一个元素为一个numpy/Tensor
        (x1,y1,x2,y2)格式的bbox
        scnensImgs：一个list，每个list中是一个numpy/Tensor格式的场景图片
        把bbox图片与所有特写进行匹配，统计bbox中特征向量被成功匹配的数量（同一向量与多张图片匹配只算一次）
        ratio是knn算法筛选用，detectThreshold代表被匹配向量占bbox中总向量数的百分比，超过该值才能算出现
        
        """
        
        #匹配器
        #flann算法匹配
        if usingFlann:
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=40)
            matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            matcher = cv2.BFMatcher(normType, crossCheck=True)
        # 匹配结果储存,若obj出现在该场景图中则对应位置元素为index，即scene中target_item的索引
        # 不存在则对应位置为-1
        res = []

        #先计算所有的特写图片的特征向量
        objInfos = []
        for obj in objs:
            keypoints = dict()
            descriptors = dict()
            #print(obj.shape)
            if imgMode == "RGB":
                keypoints["R"], descriptors["R"] = self.detectAndCompute(obj[:,:,0])
                keypoints["G"], descriptors["G"] = self.detectAndCompute(obj[:,:,1])
                keypoints["B"], descriptors["B"] = self.detectAndCompute(obj[:,:,2])
            elif imgMode == "Gray":
                t_obj = cv2.cvtColor(obj,cv2.COLOR_BGR2GRAY)
                keypoints["Gray"],descriptors["Gray"] = self.detectAndCompute(t_obj)
            elif imgMode == "HSV":
                temp_obj = cv2.cvtColor(obj,cv2.COLOR_BGR2HSV)
                h_obj, s, v = cv2.split(temp_obj)
                #仅使用H通道
                keypoints["H"],descriptors["H"] = self.detectAndCompute(h_obj)
                print(h_obj)
                print(descriptors["H"].shape)
            else:
                raise NotImplementedError
                
            objInfos.append([keypoints, descriptors])

        MaxPointsEachScene = np.zeros(len(scenesImgs))
        for i,sceneImg in enumerate(scenesImgs):
            
            bboxes = sceneBboxes[i]
            target_items = boxes.get_targetItems_by_boxes(sceneImg, np.array(bboxes).reshape(-1,4))
            #print(sceneImg.shape)
            #print(bboxes)
            if bboxes.size == 0:
                print("scene id:{} no targets detected\n".format(i))
                res.append(-1)
                continue

            #记录每个box被匹配的特征向量的最大数量
            PointsMatchedForEachBox = np.zeros(bboxes.shape[0])
            #记录每个box特征向量的数量
            PointsForEachBox = np.zeros(bboxes.shape[0])
            #依次匹配obj的特征向量与scene中检测的target的特征向量
            for index in range(bboxes.shape[0]):
                target_item = target_items[index]
                #print(bboxes[index])
                #print(target_item.shape)
                keypoints = dict()
                descriptors = dict()
                #for key in objInfos[0][0].keys():
                if imgMode == "RGB":
                    keypoints["R"], descriptors["R"] = self.detectAndCompute(target_item[:,:,0])
                    keypoints["G"], descriptors["G"] = self.detectAndCompute(target_item[:,:,1])
                    keypoints["B"], descriptors["B"] = self.detectAndCompute(target_item[:,:,2])
                elif imgMode == "Gray":
                    t_target_item = cv2.cvtColor(target_item,cv2.COLOR_BGR2GRAY)
                    keypoints["Gray"],descriptors["Gray"] = self.detectAndCompute(t_target_item)
                elif imgMode == "HSV":
                    temp_target_item = cv2.cvtColor(target_item,cv2.COLOR_BGR2HSV)
                    h_target, s, v = cv2.split(temp_target_item)
                #仅使用H通道
                    keypoints["H"],descriptors["H"] = self.detectAndCompute(h_target)
                    print(descriptors["H"].shape)
                    
                #记录每个通道中成功匹配的特征向量
                PointsMatched = [[]for i in range(len(objInfos[0][0].keys()))]
                #每张特写图片
                for objInfo in objInfos:
                    matchesCount = 0
                    for channel,key in enumerate(objInfo[0].keys()):
                        matches = matcher.knnMatch(descriptors[key], objInfo[1][key],k=2)
                        #累加得到特征向量个数
                        if descriptors[key] is None:
                            PointsForEachBox[index] += 0
                        else:
                            PointsForEachBox[index] += len(descriptors[key])
                        
                        for m,n in matches:
                            #sift距离匹配的阈值
                            if m.distance < ratio * n.distance:
                                #记录场景图中被匹配的特征向量
                                if not m.queryIdx in PointsMatched[channel]:
                                    PointsMatched[channel].append(m.queryIdx)
                
                #bboox中图片特征点被匹配的数量
                PointsMatchedForEachBox[index] = 0
                for DescripterIdsInEachChannel in PointsMatched:
                    PointsMatchedForEachBox[index] += len(DescripterIdsInEachChannel)
            
            MaxPointsEachScene[i] = PointsMatchedForEachBox.max()
            
            BestMatchBboxId = PointsMatchedForEachBox.argmax()
            
            #print("第{}号场景，最大被匹配向量数：{}，总向量数：{}".format(i,PointsMatchedForEachBox[BestMatchBboxId],PointsForEachBox[BestMatchBboxId]))
            
            if PointsForEachBox[BestMatchBboxId] == 0:
                res.append(-1)
            #匹配的数量要大于阈值才认定为出现
            #if PointsMatchedForEachBox[BestMatchBboxId]/PointsForEachBox[BestMatchBboxId] >= detectThreshold:
            if PointsMatchedForEachBox[BestMatchBboxId] > detectThreshold:
                res.append(BestMatchBboxId)
            else:
                res.append(-1)

        return np.array(res),MaxPointsEachScene
    


if __name__ == '__main__':
    #加载数据集
    print("lodaing DataSet")
    start = time.time()

    dataSet = myDataSet()
    testMugs, testScenes3, testScenes5 = dataSet.getDataSet("test")
    print(testMugs[1],testScenes3[0])
    objImgs,SceneImgs = dataSet.ReadPictures(testMugs[1],testScenes3[0])
    label = testMugs[1][-2:]

    end = time.time()
    print("time usage:{:.2f}s".format(end-start))
    #目标检测预处理
    print("generationg Bboxes")
    start = time.time()
    sceneBboxes = []
    mydet = detecter()
    scoreThreshold = 0.1
    maxCandidates = 10
    for SceneImg in SceneImgs:
        #目标检测，保留大于detectThreshold和数量小于等于maxCandidateItems的结果
        bboxes,scores = mydet.detect_sortedByScore(SceneImg, "cup")
        bboxes,scores = detecter.filtWithScores(bboxes,scores,scoreThreshold,maxCandidates)
        sceneBboxes.append(bboxes)

    end = time.time()
    print("time usage:{:.2f}s".format(end-start))
    #SIFT匹配
    print("doing SIFT")
    start = time.time()
    points_threshold = 30
    pictures_threshold = 3
    normType = cv2.NORM_L2
    mySiftMatcher = SIFT_matcher(points_threshold = points_threshold, pictures_threshold = pictures_threshold)
    bestMatches = mySiftMatcher.MatchObjInSecene(objImgs,sceneBboxes,SceneImgs,normType)

    end = time.time()
    print("time usage:{:.2f}s".format(end-start))

    print("evaluating")
    start = time.time()
    #评价指标
    annotationData = dataSet.GetAnnotation(testScenes3[0])
    iouThreshold = 0.9
    #testMugs[1]是mug_14,testScenes3[0]是scene_23,其中含有mug_14
    isPositiveSample = True
    tp,fp,tn,fn,fm = evaluater.evaluateOneObj(label, sceneBboxes, bestMatches, annotationData)

    end = time.time()
    print("time usage:{:.2f}s".format(end-start))
    print(tp,fp,tn,fn,fm)