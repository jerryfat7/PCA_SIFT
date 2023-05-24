from PCA_matcher import PCA_matcher
import numpy as np
import cv2
import boxes
#import torch
#from SIFT_matcher import SIFT_matcher
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

    mode = "test"    #"train", "val", "test"

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
    #在哪个数据集的场景中匹配
    MugsSet, Scenes3Set, Scenes5Set = dataSet.getDataSet(mode)
    ScenesSet = Scenes3Set + Scenes5Set
    #使用训练集中的杯子生成特征杯子空间
    trainMugs ,_,__ = dataSet.getDataSet("train")

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
    PositiveResult["minDistances"] = []

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

    #读取所有的特写照片
    objImgs = []
    for mugsFolder in trainMugs:
        objImgs += dataSet.ReadMugsPictures(mugsFolder)
    #SceneImgs = dataSet.ReadScenePictures(trainScenes3[1])
    myLogger.info("训练集所用特写数量：{}".format(len(objImgs)))
    
    #目标检测预处理
    myLogger.info("generationg Bboxes for PCA train Sets")
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
    #PCA参数
    PCA_distance_threshold = 1.5e4
    norm = 2
    n_components = 40
    normalize = PCA_matcher.Default_normalize
    #训练PCA模型
    pcaModel = PCA_matcher(n_components)
    pcaModel.train(newObjImgs, normalize)

    end = time.time()
    myLogger.info("time usage:{:.2f}s".format(end-start))

    #训练完毕，在该PCA模型张成的子空间上进行匹配

    for SceneFolder in ScenesSet:
        SceneImgs = dataSet.ReadScenePictures(SceneFolder)

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
            
            newObjImgs = []
            for ObjImg in objImgs:
                #目标检测，保留大于detectThreshold和数量小于等于maxCandidateItems的结果
                bboxes,scores = mydet.detect_sortedByScore(ObjImg, "cup")
                assert bboxes.shape[0] > 0
                cup_bbox = bboxes[0,:]

                newObjImg = boxes.get_targetItems_by_boxes(ObjImg, cup_bbox.reshape(-1,4))[0]
                newObjImgs.append(PCA_matcher.Default_normalize(newObjImg)) 
            
            objImgs = newObjImgs
            
            
            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))

            myLogger.info("doing PCA match")
            start = time.time()

            #模型推断匹配
            bestMatches, minDistances = pcaModel.MatchObjInSecene(objImgs, sceneBboxes, SceneImgs, PCA_distance_threshold, norm, normalize)


            end = time.time()
            myLogger.info("time usage:{:.2f}s".format(end-start))

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
            if isPositiveSample:
                myLogger.warning("是正样本")
            else:
                myLogger.warning("是负样本")
            #print("tp:{},fp-fn:{},fd:{}".format())
            myLogger.warning("tp,fp,tn,fn,fd\n"+"{} {} {} {} {}".format(tp,fp,tn,fn,fd))
            myLogger.warning("DenyClassification, DenyClassificationWithDetection\n"+"{} {}".format(DenyClassification, DenyClassificationWithDetection))
            np.save(r".\result\{}+{}".format(mugsFolder,SceneFolder), minDistances)

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
    # 储存每个最优匹配的最低距离
    P_toSave = np.stack(PositiveResult["minDistances"])
    N_toSave = np.stack(NegativeResult["minDistances"])

    #根据最短距离进行SVM分类，提供一个建议分割面
    #首先平衡正负样本
    po,ne = classificationBoundary.balancePositiveAndNegative(P_toSave, N_toSave, 1)
    suggestedBound = classificationBoundary.getClassifyPointSVM(po,ne)
    myLogger.warning("SVM计算的最优分割面应为：{}".format(suggestedBound))

    np.save(r".\result\positive", P_toSave)
    np.save(r".\result\negative", N_toSave)

    # 保存预识别的内容
    data_write = json.dumps(preDetectData,indent=4)
    with open(jsonPath, 'w') as f_json:
        f_json.write(data_write)