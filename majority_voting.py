from SIFT_matcher import SIFT_matcher
from PCA_matcher import PCA_matcher
import numpy as np
import cv2
import boxes
#import torch
#from SIFT_matcher import SIFT_matcher
#计算投票
from scipy import stats

import classificationBoundary
#for test
from mylogging import myLogger
from PIL import Image
from detecter import detecter
from myDataSet import myDataSet
from evaluater import evaluateOneObj
import time

import os
import os.path as osp
import json


if __name__ == '__main__':
    #三个模型的参数
    SIFT_params_1 = dict()
    SIFT_params_1["MatchesNumPrior"] = True
    SIFT_params_1["pointsThreshold"] = 7
    SIFT_params_1["picturesThreshold"] = 1
    SIFT_params_1["edgeThreshold"] = 11
    SIFT_params_1["normType"] = cv2.NORM_L2
    SIFT_params_1["ratio"] = 0.7
    SIFT_params_1["contrastThreshold"] = 0.03
    SIFT_params_1["sigma"] = 3
    SIFT_params_1["usingFlann"] = True
    SIFT_params_1["imgMode"] = "RGB"
    sift1 = cv2.xfeatures2d.SIFT_create(sigma = SIFT_params_1["sigma"]\
                                               ,contrastThreshold = SIFT_params_1["contrastThreshold"]\
                                                ,edgeThreshold = SIFT_params_1["edgeThreshold"])
    mySiftMatcher1 = SIFT_matcher(sift=sift1)

    # SIFT_params_2 = dict()
    # SIFT_params_2["MatchesNumPrior"] = True
    # SIFT_params_2["pointsThreshold"] = 5
    # SIFT_params_2["picturesThreshold"] = 1
    # SIFT_params_2["edgeThreshold"] = 11
    # SIFT_params_2["normType"] = cv2.NORM_L2
    # SIFT_params_2["ratio"] = 0.65
    # SIFT_params_2["contrastThreshold"] = 0.03
    # SIFT_params_2["sigma"] = 2.5
    # SIFT_params_2["usingFlann"] = True
    # SIFT_params_2["imgMode"] = "RGB"

    # sift2 = cv2.xfeatures2d.SIFT_create(sigma = SIFT_params_2["sigma"]\
    #                                            ,contrastThreshold = SIFT_params_2["contrastThreshold"]\
    #                                             ,edgeThreshold = SIFT_params_2["edgeThreshold"])
    # mySiftMatcher2 = SIFT_matcher(sift=sift2)

    PCA_params = dict()
    PCA_params["PCA_distance_threshold"] = 1.4e4
    PCA_params["norm"] = 2
    PCA_params["n_components"] = 38

    PCA1_params = dict()
    PCA1_params["PCA_distance_threshold"] = 1.1e4
    PCA1_params["norm"] = 2
    PCA1_params["n_components"] = 35
    #PCA归一化函数
    PCA_normalize = PCA_matcher.Default_normalize

    #选择哪个数据集进行验证
    mode = "test"    #"train", "val", "test"

    #加载已预识别的方框，如不需要预加载，删除文件或文件中对应场景的方框即可
    jsonPath = mode + ".json"
    if osp.exists(jsonPath):
        myLogger.info("using pre-detect boxes")
        with open(jsonPath) as json_file:
            preDetectData = json.load(json_file)
            preLoad = True
    else:
        myLogger.info("no pre-detect boxes, generate detect boxes realtime")
        preDetectData = dict()
        preLoad = False
    
    #加载数据集
    myLogger.info("lodaing DataSet")
    #root = r"D:\学习\大四下\模式识别\大作业_瓶子检测\miniTest"
    root = r"D:\学习\大四下\模式识别\大作业_瓶子检测\Personalized_Segmentation"
    dataSet = myDataSet(root)
    #dataSet = myDataSet()
    MugsSet, Scenes3Set, Scenes5Set = dataSet.getDataSet(mode)
    ScenesSet = Scenes3Set + Scenes5Set

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
    #记录PCA的最短距离用
    PositiveResult["minDistances"] = []
    PositiveResult["minDistances1"] = []
    PositiveResult["SIFT1_maxMatchPoints"] = []
    #PositiveResult["SIFT2_maxMatchPoints"] = []

    NegativeResult = dict()
    NegativeResult["tp"] = 0
    NegativeResult["fp"] = 0
    NegativeResult["tn"] = 0
    NegativeResult["fn"] = 0
    NegativeResult["DC"] = 0
    NegativeResult["DCWD"] = 0
    NegativeResult["fd"] = 0
    NegativeResult["count"] = 0
    NegativeResult["minDistances"] = []
    NegativeResult["minDistances1"] = []
    NegativeResult["SIFT1_maxMatchPoints"] = []
    #NegativeResult["SIFT2_maxMatchPoints"] = []

    #******************************************训练PCA
    #使用训练集中的杯子生成特征杯子空间
    trainMugs ,_,__ = dataSet.getDataSet("train")
    #读取所有的特写照片
    objImgs = []
    for mugsFolder in trainMugs:
        objImgs += dataSet.ReadMugsPictures(mugsFolder)
    #SceneImgs = dataSet.ReadScenePictures(trainScenes3[1])
    myLogger.info("训练集所用特写数量：{}".format(len(objImgs)))

    newObjImgs = []
    for ObjImg in objImgs:
        #目标检测，保留大于detectThreshold和数量小于等于maxCandidateItems的结果
        bboxes,scores = mydet.detect_sortedByScore(ObjImg, "cup")
        assert bboxes.shape[0] > 0
        cup_bbox = bboxes[0,:]

        newObjImg = boxes.get_targetItems_by_boxes(ObjImg, cup_bbox.reshape(-1,4))[0]
        
        newObjImgs.append(newObjImg)

    start = time.time()
    myLogger.info("training PCA")
    #训练PCA模型
    PCA_Model = PCA_matcher(PCA_params["n_components"])
    PCA_Model.train(newObjImgs, PCA_normalize)

    PCA_Model1 = PCA_matcher(PCA1_params["n_components"])
    PCA_Model1.train(newObjImgs, PCA_normalize)
    
    end = time.time()
    myLogger.info("time usage:{:.2f}s".format(end-start))
    #*******************************************************************

    for SceneFolder in ScenesSet:
        SceneImgs = dataSet.ReadScenePictures(SceneFolder)
        #无预加载数据
        if not preLoad or preDetectData.get(SceneFolder) is None:
            #目标检测预处理
            myLogger.info("generationg Scene Bboxes")
            start = time.time()
            #场景杯子检测
            
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

            preDetectData[SceneFolder] = [sceneBbox.tolist() for sceneBbox in sceneBboxes]
            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))
        else:
            sceneBboxes = [np.array(sceneBbox) for sceneBbox in preDetectData[SceneFolder]]

        for mugsFolder in MugsSet:
            #print(testMugs[1],testScenes3[0])
            myLogger.warning("\nmugs:{} Scene:{}".format(mugsFolder, SceneFolder))
            #objImgs,SceneImgs = dataSet.ReadPictures(testMugs[1],testScenes3[0])
            
            objImgs = dataSet.ReadMugsPictures(mugsFolder)
            
            #label = testMugs[1][-2:]
            label = mugsFolder[-2:]
            myLogger.info("label:{}".format(label))
            #目标检测预处理
            myLogger.info("特写预处理")
            start = time.time()
            
            #特写预处理
            #SIFT预处理会在match中进行，PCA的obj需要提前进行预处理
            SIFT_ObjImgs = []
            PCA_ObjImgs = []
            for ObjImg in objImgs:
                #目标检测，保留大于detectThreshold和数量小于等于maxCandidateItems的结果
                bboxes,scores = mydet.detect_sortedByScore(ObjImg, "cup")
                assert bboxes.shape[0] > 0
                cup_bbox = bboxes[0,:]

                newObjImg = boxes.get_targetItems_by_boxes(ObjImg, cup_bbox.reshape(-1,4))[0]
                SIFT_ObjImgs.append(newObjImg)
                PCA_ObjImgs.append(PCA_normalize(newObjImg))
            

            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))
        
            # sift1
            #******************************************************************
            myLogger.info("doing SIFT 1")
            start = time.time()

            bestMatches1,MaxMatchesCount1 = mySiftMatcher1.MatchObjInSeceneUsingMaxMatches(SIFT_ObjImgs,sceneBboxes,\
                                                        SceneImgs,SIFT_params_1["pointsThreshold"],SIFT_params_1["picturesThreshold"],\
                                                            SIFT_params_1["ratio"],SIFT_params_1["normType"],\
                                                        SIFT_params_1["MatchesNumPrior"],SIFT_params_1["usingFlann"], SIFT_params_1["imgMode"])
            
            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))
            #******************************************************************

            # sift2
            #******************************************************************
            # myLogger.info("doing SIFT 2")
            # start = time.time()

            # bestMatches2,MaxMatchesCount2 = mySiftMatcher2.MatchObjInSeceneUsingMaxMatches(SIFT_ObjImgs,sceneBboxes,\
            #                                             SceneImgs,SIFT_params_2["pointsThreshold"],SIFT_params_2["picturesThreshold"],\
            #                                                 SIFT_params_2["ratio"],SIFT_params_2["normType"],\
            #                                             SIFT_params_2["MatchesNumPrior"],SIFT_params_2["usingFlann"], SIFT_params_2["imgMode"])
            
            # end = time.time()
            # myLogger.info("time usage:{:.2f}s".format(end-start))
            #******************************************************************

            # PCA
            #*******************************************************
            myLogger.info("doing PCA 0")
            start = time.time()

            bestMatches_PCA, minDistances = PCA_Model.MatchObjInSecene(PCA_ObjImgs, sceneBboxes, SceneImgs, PCA_params["PCA_distance_threshold"], PCA_params["norm"], PCA_normalize)

            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))

            #*******************************************************

            # PCA1
            #*******************************************************
            myLogger.info("doing PCA 1")
            start = time.time()

            bestMatches_PCA1, minDistances1 = PCA_Model1.MatchObjInSecene(PCA_ObjImgs, sceneBboxes, SceneImgs, PCA1_params["PCA_distance_threshold"], PCA1_params["norm"], PCA_normalize)

            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))

            #*******************************************************

            #bestMatches_total = np.stack((bestMatches_PCA, bestMatches1, bestMatches2))
            bestMatches_total = np.stack((bestMatches_PCA, bestMatches1, bestMatches_PCA1))
            bestMatches = stats.mode(bestMatches_total,keepdims=True)[0][0]

            myLogger.info("evaluating")
            
            #评价指标
            annotationData = dataSet.GetAnnotation(SceneFolder)
            iouThreshold = 0.9
            #判断是否为正样本（该场景中有杯子出现）
            isPositiveSample = dataSet.isPositiveSample(mugsFolder, SceneFolder)

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
            recordDict["fn"] += fn
            recordDict["DC"] += DenyClassification
            recordDict["DCWD"] += DenyClassificationWithDetection
            recordDict["fd"] += fd
            recordDict["count"] += 1
            recordDict["minDistances"].append(minDistances)
            recordDict["minDistances1"].append(minDistances1)
            recordDict["SIFT1_maxMatchPoints"].append(MaxMatchesCount1)
            #recordDict["SIFT2_maxMatchPoints"].append(MaxMatchesCount2)
            if isPositiveSample:
                myLogger.warning("是正样本")
            else:
                myLogger.warning("是负样本")
            #print("tp:{},fp-fn:{},fd:{}".format())
            myLogger.warning("tp,fp,tn,fn,fd\n"+"{} {} {} {} {}".format(tp,fp,tn,fn,fd))
            myLogger.warning("DenyClassification, DenyClassificationWithDetection\n"+"{} {}".format(DenyClassification, DenyClassificationWithDetection))
            #np.save(r".\result\{}+{}".format(mugsFolder,SceneFolder), minDistances)

    #正样本分类准确率=正确识别到杯子的次数/目标检测模型识别到目标杯子框，且SIFT不识别为不存在的次数
    #fn-fp是错误地识别为不存在的次数，因为如果存在目标杯子，则fp和fn应始终相等，存在但是未识别会导致fp单增
    #存在但是识别错误fp和fn均增加
    #不存在但是识别到会导致fn单增，但由于是正样本，几乎不存在此中情况（绝大部分图片中目标杯子均出现）
    pos_classify_acc = PositiveResult["tp"] / (100 * PositiveResult["count"] - \
                                               PositiveResult["DCWD"] - \
                                                (PositiveResult["fn"] - PositiveResult["fp"]\
                                                 - PositiveResult["fd"]))
    #目标检测模型正样本拒绝错误率=存在目标杯子但是目标检测模型未识别的次数
    pos_detctor_false_detect_rate = PositiveResult["fd"] / (100 * PositiveResult["count"])

    #PCA模型正样本拒绝错误率=标检测模型识别到目标杯子框，但是SIFT模型未识别的次数
    pos_PCA_false_detect_rate = (PositiveResult["fn"] - PositiveResult["fp"])/ (100 * PositiveResult["count"])
    #负样本拒绝准确率
    neg_reject_acc = 1 - NegativeResult["fp"] / (100 * NegativeResult["count"])

    myLogger.warning("正样本分类准确率:{:.3f}".format(pos_classify_acc))
    myLogger.warning("目标检测模型正样本拒绝错误率:{:.3f}".format(pos_detctor_false_detect_rate))
    myLogger.warning("PCA模型正样本拒绝错误率:{:.3f}".format(pos_PCA_false_detect_rate))
    myLogger.warning("负样本拒绝准确率:{:.3f}".format(neg_reject_acc))

    myLogger.warning(PositiveResult)
    myLogger.warning(NegativeResult)
    # PCA最短距离
    P_toSave = np.stack(PositiveResult["minDistances"])
    N_toSave = np.stack(NegativeResult["minDistances"])
    P_toSave1 = np.stack(PositiveResult["minDistances1"])
    N_toSave1 = np.stack(NegativeResult["minDistances1"])
    #两个SIFT
    SIFT1_P = np.stack(PositiveResult["SIFT1_maxMatchPoints"])
    SIFT1_N = np.stack(NegativeResult["SIFT1_maxMatchPoints"])
    #SIFT2_P = np.stack(PositiveResult["SIFT2_maxMatchPoints"])
    #SIFT2_N = np.stack(NegativeResult["SIFT2_maxMatchPoints"])

    po,ne = classificationBoundary.balancePositiveAndNegative(P_toSave, N_toSave, 1)
    suggestedBound = classificationBoundary.getClassifyPointSVM(po,ne)
    myLogger.warning("SVM计算的PCA最优分割面应为：{}".format(suggestedBound))

    po,ne = classificationBoundary.balancePositiveAndNegative(P_toSave1, N_toSave1, 1)
    suggestedBound = classificationBoundary.getClassifyPointSVM(po,ne)
    myLogger.warning("SVM计算的PCA 1最优分割面应为：{}".format(suggestedBound))

    po,ne = classificationBoundary.balancePositiveAndNegative(SIFT1_P, SIFT1_N, 1)
    suggestedBound = classificationBoundary.getClassifyPointSVM(po,ne)
    myLogger.warning("SVM计算的SIFT1最优分割面应为：{}".format(suggestedBound))

    # po,ne = classificationBoundary.balancePositiveAndNegative(SIFT2_P, SIFT2_N, 1)
    # suggestedBound = classificationBoundary.getClassifyPointSVM(po,ne)
    # myLogger.warning("SVM计算的SIFT2最优分割面应为：{}".format(suggestedBound))

    np.save(r".\result\PCA_positive", P_toSave)
    np.save(r".\result\PCA_negative", N_toSave)
    np.save(r".\result\PCA_positive1", P_toSave1)
    np.save(r".\result\PCA_negative1", N_toSave1)
    np.save(r".\result\SIFT1_positive", SIFT1_P)
    np.save(r".\result\SIFT1_negative", SIFT1_N)
    #np.save(r".\result\SIFT2_positive", SIFT2_P)
    #np.save(r".\result\SIFT2_negative", SIFT2_N)

    # 保存预识别的内容
    data_write = json.dumps(preDetectData,indent=4)
    with open(jsonPath, 'w') as f_json:
        f_json.write(data_write)



