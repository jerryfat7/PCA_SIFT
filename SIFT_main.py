from SIFT_matcher import SIFT_matcher
import numpy as np
import cv2
import boxes
#import torch
#from SIFT_matcher import SIFT_matcher

#for test
from mylogging import myLogger
from PIL import Image
from detecter import detecter
from myDataSet import myDataSet
from evaluater import evaluateOneObj
import time

if __name__ == '__main__':

    funcs = ["MaxMatch", "MaxMatchV1", "MaxPoints"]
    func = funcs[0]
    mode = "train"    #"train", "val", "test"
    #加载数据集
    myLogger.info("lodaing DataSet")
    #root = r"D:\学习\大四下\模式识别\大作业_瓶子检测\miniTest"
    root = r"D:\学习\大四下\模式识别\大作业_瓶子检测\Personalized_Segmentation"
    dataSet = myDataSet(root)
    #dataSet = myDataSet()
    MugsSet, Scenes3Set, Scenes5Set = dataSet.getDataSet("train")

    #目标检测参数
    mydet = detecter()
    scoreThreshold = 0.15
    maxCandidates = 10
    #筛去过小的目标
    minSize = 10

    #记录总体数据
    PositiveResult = dict()
    PositiveResult["tp"] = 0
    PositiveResult["fp"] = 0
    PositiveResult["tn"] = 0
    PositiveResult["fn"] = 0
    PositiveResult["DC"] = 0
    PositiveResult["DCWD"] = 0
    PositiveResult["fd"] = 0
    PositiveResult["count"] = 0
    PositiveResult["maxMatchPoints"] = []

    NegativeResult = dict()
    NegativeResult["tp"] = 0
    NegativeResult["fp"] = 0
    NegativeResult["tn"] = 0
    NegativeResult["fn"] = 0
    NegativeResult["DC"] = 0
    NegativeResult["DCWD"] = 0
    NegativeResult["fd"] = 0
    NegativeResult["count"] = 0
    NegativeResult["maxMatchPoints"] = []

    
    for SceneFolder in Scenes3Set:
        #目标检测预处理
        myLogger.info("generationg Scene Bboxes")
        start = time.time()
        #场景杯子检测
        SceneImgs = dataSet.ReadScenePictures(SceneFolder)
        sceneBboxes = []

        for SceneImg in SceneImgs:
            #目标检测，保留大于detectThreshold和数量小于等于maxCandidateItems的结果
            bboxes,scores = mydet.detect_sortedByScore(SceneImg, "cup")
            
            bboxes,scores = detecter.filtWithScores(bboxes,scores,scoreThreshold,maxCandidates)
            bboxes = bboxes.reshape(-1,4)
            #print(bboxes.shape)
            #print(scores.shape)
            bboxes = boxes.remove_small_boxes(bboxes, minSize)
            #print(bboxes.shape)
            sceneBboxes.append(bboxes)

        end = time.time()
        myLogger.info("time usage:{:.2f}s".format(end-start))

        for mugsFolder in MugsSet:
            #print(testMugs[1],testScenes3[0])
            myLogger.warning("mugs:{} Scene:{}".format(mugsFolder, SceneFolder))
            #objImgs,SceneImgs = dataSet.ReadPictures(testMugs[1],testScenes3[0])
            
            objImgs = dataSet.ReadMugsPictures(mugsFolder)
            
            #label = testMugs[1][-2:]
            label = mugsFolder[-2:]
            myLogger.info("label:{}".format(label))
            #目标检测预处理
            myLogger.info("特写预处理")
            start = time.time()
            
            #特写预处理
            
            newObjImgs = []
            for ObjImg in objImgs:
                #目标检测，保留大于detectThreshold和数量小于等于maxCandidateItems的结果
                bboxes,scores = mydet.detect_sortedByScore(ObjImg, "cup")
                assert bboxes.shape[0] > 0
                cup_bbox = bboxes[0,:]

                newObjImg = boxes.get_targetItems_by_boxes(ObjImg, cup_bbox.reshape(-1,4))[0]
                newObjImgs.append(newObjImg)
                
            objImgs = newObjImgs
            
            
            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))

            myLogger.info("doing SIFT")
            start = time.time()

            if func == "MaxMatch":
                MatchesNumPrior = True
                pointsThreshold = 10
                picturesThreshold = 1
                edgeThreshold = 11
                normType = cv2.NORM_L2
                ratio = 0.7
                contrastThreshold = 0.03
                sigma = 3
                usingFlann = True
                imgMode = "RGB"

                myLogger.warning("MatchesNumPrior,pointsThreshold,picturesThreshold,normType,imgMode,ratio,sigma,contrastThreshold,edgeThreshold")
                myLogger.warning("参数：{}，{}，{}，{}，{}，{}，{}，{}，{}".format(MatchesNumPrior,pointsThreshold,\
                    picturesThreshold,normType,imgMode,ratio,sigma,contrastThreshold,edgeThreshold))

                sift = cv2.xfeatures2d.SIFT_create(sigma = sigma, contrastThreshold = contrastThreshold,edgeThreshold = edgeThreshold)
                mySiftMatcher = SIFT_matcher(sift=sift)

                bestMatches,MaxMatchesCount = mySiftMatcher.MatchObjInSeceneUsingMaxMatches(objImgs,sceneBboxes,\
                                                            SceneImgs,pointsThreshold,picturesThreshold,ratio,normType,\
                                                            MatchesNumPrior,usingFlann, imgMode)

            elif func == "MaxMatchV1":
                MatchesNumPrior = False
                pointsThreshold = 8
                picturesThreshold = 1
                edgeThreshold = 11
                normType = cv2.NORM_L2
                ratio = 0.7
                contrastThreshold = 0.02
                sigma = 1
                usingFlann = True
                imgMode = "RGB"

                myLogger.warning("MatchesNumPrior,pointsThreshold,picturesThreshold,normType,imgMode,ratio,sigma,contrastThreshold,edgeThreshold")
                myLogger.warning("参数：{}，{}，{}，{}，{}，{}，{}，{}，{}".format(MatchesNumPrior,pointsThreshold,\
                    picturesThreshold,normType,imgMode,ratio,sigma,contrastThreshold,edgeThreshold))

                sift = cv2.xfeatures2d.SIFT_create(sigma = sigma, contrastThreshold = contrastThreshold,edgeThreshold = edgeThreshold)
                mySiftMatcher = SIFT_matcher(sift=sift)

                bestMatches,MaxMatchesCount = mySiftMatcher.MatchObjInSeceneUsingMaxMatches_v1(objImgs,sceneBboxes,\
                                                            SceneImgs,pointsThreshold,picturesThreshold,ratio,normType,\
                                                            MatchesNumPrior,usingFlann, imgMode)

            elif func == "MaxPoints":
                sigma = 3
                contrastThreshold = 0.03
                edgeThreshold = 11
                ratio = 0.7
                #detectThreshold = 0.01
                detectThreshold = 8
                normType = cv2.NORM_L2
                imgMode = "RGB"
                usingFlann = True

                myLogger.info("detectThreshold，normType,imgMode,ratio,sigma,contrastThreshold,edgeThreshold")
                myLogger.info("参数：{}，{}，{}，{}，{}，{}，{}".format(detectThreshold, normType,imgMode,ratio,sigma,contrastThreshold,edgeThreshold))
                sift = None
                sift = cv2.xfeatures2d.SIFT_create(sigma = sigma, contrastThreshold = contrastThreshold, edgeThreshold = edgeThreshold)

                mySiftMatcher = SIFT_matcher(sift=sift)

                bestMatches,MaxMatchesCount = mySiftMatcher.MatchObjInSeceneUsingMaxPoints(objImgs,sceneBboxes,\
                                                            SceneImgs,ratio,normType,detectThreshold,\
                                                            usingFlann, imgMode)
                #print(bestMatches)

            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))

            myLogger.info("evaluating")
            #评价指标
            annotationData = dataSet.GetAnnotation(SceneFolder)
            iouThreshold = 0.9
            #判断是否为正样本（该场景中有杯子出现）
            isPositiveSample = dataSet.isPositiveSample(mugsFolder, SceneFolder)

            #testMugs[1]是mug_14,testScenes3[0]是scene_23,其中含有mug_14
            tp,fp,tn,fn,DenyClassification, DenyClassificationWithDetection, fd = evaluateOneObj(label,\
                                            sceneBboxes, bestMatches,\
                                            annotationData,iouThreshold,isPositiveSample)
            
            
            if isPositiveSample:
                recordDict = PositiveResult
            else:
                recordDict = NegativeResult

            recordDict["tp"] += tp
            recordDict["fp"] += fp
            recordDict["tn"] += tn
            recordDict["DC"] += DenyClassification
            recordDict["DCWD"] += DenyClassificationWithDetection
            recordDict["fd"] += fd
            recordDict["count"] += 1
            recordDict["maxMatchPoints"].append(MaxMatchesCount)
            if isPositiveSample:
                myLogger.warning("是正样本")
            else:
                myLogger.warning("是负样本")
            #print("tp:{},fp-fn:{},fd:{}".format())
            myLogger.warning("tp,fp,tn,fn,fd\n"+"{} {} {} {} {}".format(tp,fp,tn,fn,fd))
            myLogger.warning("DenyClassification, DenyClassificationWithDetection\n"+"{} {}".format(DenyClassification, DenyClassificationWithDetection))
            np.save(r".\result\{}+{}".format(mugsFolder,SceneFolder), MaxMatchesCount)

    myLogger.warning(PositiveResult)
    myLogger.warning(NegativeResult)
    # 
    P_toSave = np.stack(PositiveResult["maxMatchPoints"])
    N_toSace = np.stack(NegativeResult["maxMatchPoints"])

    np.save(r".\result\postive", P_toSave)
    np.save(r".\result\negative", N_toSace)