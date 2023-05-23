import boxes
import numpy as np
from det_utils import Matcher

import torch
from torch import Tensor


def evaluateOneObj(label, sceneBboxes, bestMatches, gtDict, iouThreshold, isPositiveSample = True):
    """
    评价这样特定物品检测这个问题，可以把它看成一个二分类问题，因为前端的目标检测网络已经确定
    故实际上算法是评判场景中截取的一个杯子是否为目标杯子。
    还需要统计成功拒绝分类的数目（有目标检测网络，包含检测不到任何杯子的情况）DenyClassificationWithDetection
    以及在有目标检测结果的情况下拒绝分类的数目DenyClassification
    但是，有前端目标检测网络阈值过高，导致杯子未被检出的可能，因此，需要计算
    杯子未检出的次数fd(false detect),若fd过高则需要降低目标检测的阈值了

    所以，返回值为tp,fp,tn,fn，DenyClassification,DenyClassificationWithDetection,fd
    便于后续进行acc，recall等计算

    input:
    label：待查找的obj的label

    sceneBboxes：detecter在scene图片中生成的bboxes，是很多杯子的bbox
    是一个list，其中每一个元素为一张图片中一个numpy/Tensor(shape = [N,4]) (x1,y1,x2,y2)格式的bbox
    bestMatches：MatchObjInSecene()函数返回的最佳匹配array，每个数字对应sceneBboxes[i]中bbox的
    索引，-1代表不存在

    redDict: 一个字典，每一个键都是一张图片的名称，值是一个列表，列表中每一个元素都是一个
    dict，每个dict里中有"label"和"bbox"两个键，分别对应预测类别和预测框的位置(x1, y1,x2,y2)
    
    iouThreshold：iou高于iouThreshold的都是正常的预测与gt的候选匹配
    
    isPositiveSample：该物品存在于该场景中，则需要分析tp,fp,tn,fn
    如果该物品不存在于该场景中，则只需分析fp就可以了，余下的都是tn

    return:
    tp,fp,tn,fn,DenyClassification, DenyClassificationWithDetection, fd
    """
    if not isPositiveSample:
        fp = np.where(bestMatches >=0 , 1, 0).sum()
        
        NoCupDetected = 0
        total = 0
        for bbox in sceneBboxes:
            total += bbox.shape[0]
            if bbox.shape[0] == 0:
                NoCupDetected += 1
        
        DenyClassificationWithDetection = len(bestMatches) - fp
        DenyClassification = DenyClassificationWithDetection - NoCupDetected
        
        
        tn = total - fp
        return 0,fp,tn,0,DenyClassification,DenyClassificationWithDetection,0
    
    tp = 0  #正样本匹配正确
    fp = 0  #正样本匹配错误
    tn = 0  #负样本匹配正确
    fn = 0  #负样本匹配错误
    DenyClassificationWithDetection = 0 #有目标检测网络，成功拒绝分类（+不存在目标的情况）
    DenyClassification = 0 #在有目标的基础上拒绝分类
    fd = 0  #目标检测未检测出杯子的次数


    intLabel = int(label)
    for i in range(len(sceneBboxes)):
        gtInfo = gtDict[list(gtDict.keys())[i]]
        
        #首先根据gt判断这张图中是否存在目标杯子
        gtBox = None
        for labelInfo in gtInfo:
            if intLabel == int(labelInfo["label"]):
                gtBox = np.array(labelInfo["bbox"])
                break
        
        if not gtBox is None:
            #若压根没有检测到任何杯子，fd+=1
            if len(sceneBboxes[i]) == 0:
                fd += 1
                DenyClassificationWithDetection += 1
                print("{}号场景未检测到任何杯子".format(i))
                assert bestMatches[i] < 0
                continue
            
            
            #先计算预测的bbox中是否有与目标bbox Iou>iouThreshold的
            ious = boxes.box_iou(gtBox.reshape(-1,4), sceneBboxes[i])
            bboxMatchGt = (ious>=iouThreshold).sum() > 0

            if bestMatches[i] < 0:
                #实际存在但是未检出，可能是fd或fn，剩余为真阴或可能存在的阳性
                #若bbox中其实有与gt匹配的，但是匹配错误，fn+=1
                if bboxMatchGt:
                    tn += sceneBboxes[i].shape[0] - 2
                    fn += 1
                else:
                #若压根没有与gt匹配的，则全为tn，且fd要增加
                    tn += sceneBboxes[i].shape[0] - 1
                    fd += 1
            else:
                #取得预测的iou
                #print(i, ious.shape)
                iou = ious[0,bestMatches[i]]
                if iou >= iouThreshold:
                    #正确匹配
                    tp += 1
                    tn += sceneBboxes[i].shape[0] - 1
                else:
                    fp += 1
                    if bboxMatchGt:
                        tn += sceneBboxes[i].shape[0] - 2
                        fn += 1
                    else:
                        tn += sceneBboxes[i].shape[0] - 1
                        fd += 1
        else:
            if bestMatches[i] < 0:
                tn += sceneBboxes[i].shape[0]
                #成功拒绝分类
                DenyClassification += 1
                DenyClassificationWithDetection += 1
            else:
                fp += 1
                tn += sceneBboxes[i].shape[0] - 1

    return tp,fp,tn,fn,DenyClassification, DenyClassificationWithDetection, fd



def evaluate(predDict, gtDict, highThreshold, lowThreshold, allow_low_quality_matches=False):
    """
    input:
    predDict: 一个字典，每一个键都是一张图片的名称，值是一个列表，列表中每一个元素都是一个
    dict，每个dict里中有"label"和"bbox"两个键，分别对应预测类别和预测框的位置(x1, y1,x2,y2)
    example:
    {
        "scene_01_00000.png": [
            {
                "label": "5",
                "bbox": [
                    138.41165161132812,
                    195.0966796875,
                    342.42889404296875,
                    421.3419494628906
                ]
            },
            {
                "label": "2",
                "bbox": [
                    412.6247253417969,
                    184.39564514160156,
                    636.6557006835938,
                    403.03167724609375
                ]
            },
        ]
    }
    gtDict与predDict为相同格式的字典。
    highThreshold，lowThreshold，allow_low_quality_matches：
    详见det_utils.Matcher，iou高于highThreshold的都是正常的预测与gt的候选匹配
    当allow_low_quality_matches=True时，Matcher会为未匹配的gt选择低于highThreshold但高于lowThreshold
    的最佳匹配
    """
    accs = []
    precisions = []
    recalls = []
    detection_recalls = []
    detection_precisions = []
    for EachImg in predDict:
        info_pred = predDict[EachImg]
        info_gt = gtDict[EachImg]

        bboxes_pred = torch.zeros(len(info_pred),4)
        bboxes_gt = torch.zeros(len(info_gt),4)
        labels_pred = np.zeros(len(info_pred))
        labels_gt = np.zeros(len(info_gt))
                               
        for i in range(len(info_pred)):
            bboxes_pred[i,:] = torch.tensor(info_pred[i]["bbox"])
            labels_pred[i] = int(info_pred[i]["label"])

        for i in range(len(info_gt)):
            bboxes_gt[i,:] = torch.tensor(info_gt[i]["bbox"])
            labels_gt[i] = int(info_gt[i]["label"])

        # bboxes_pred = bboxes_pred.int()
        # bboxes_gt = bboxes_gt.int()

        ious = boxes.box_iou(bboxes_gt, bboxes_pred)
        #print(ious)
        matcher = Matcher(highThreshold, lowThreshold,allow_low_quality_matches)
        matches = matcher(ious)
        #print(matches)

        #对每个ground truth方框的预测类别
        #未识别出框的话为负数
        predLabelIngt = torch.where(matches>=0,labels_pred[matches], matches)

        #未检出的gt
        Undetected_gt = torch.where(matches>=0, 0, 1).sum()

        # gt的检出率（recall）,被检出gt占总gt的比例
        detection_recall = 1 - Undetected_gt / labels_gt.shape[0]
        detection_recalls.append(detection_recall)
        # pred的检出率（precision），被检出gt占pred的比例
        detection_precision = 1 - Undetected_gt / labels_pred.shape[0]
        detection_precisions.append(detection_precision)
        # 判定acc,错误率需要考虑不出现在gt里的bbox
        # 总数 = gt数 + pred数 - 正确识别数（交集）= Undetected_gt + pred数
        #detected_gt = labels_gt.shape[0] - Undetected_gt
        acc = (predLabelIngt == labels_gt).sum() / (Undetected_gt + labels_pred.shape[0])
        accs.append(acc)
        #recall = tp/(tp+fn)
        recall = (predLabelIngt == labels_gt).mean()
        recalls.append(recall)
        #precision = tp/(tp+fp)
        precision = (predLabelIngt == labels_gt).sum() / labels_pred.shape[0]
        precisions.append(precision)

    return 

if __name__ == '__main__':
    gt = {
    "scene_01_00000.png": [
        {
            "label": "5",
            "bbox": [
                138.41165161132812,
                195.0966796875,
                342.42889404296875,
                421.3419494628906
            ]
        },
        {
            "label": "2",
            "bbox": [
                412.6247253417969,
                184.39564514160156,
                636.6557006835938,
                403.03167724609375
            ]
        },
        {
            "label": "7",
            "bbox": [
                663.2958374023438,
                182.13267517089844,
                896.39111328125,
                404.1787109375
            ]
        }
    ],
    }

    pred = {
    "scene_01_00000.png": [
        {
            "label": "2",
            "bbox": [
                412.6247253417969,
                184.39564514160156,
                636.6557006835938,
                403.03167724609375
            ]
        },
        {
            "label": "5",
            "bbox": [
                138.41165161132812,
                195.0966796875,
                342.42889404296875,
                421.3419494628906
            ]
        },
        {
            "label": "7",
            "bbox": [
                100,
                100,
                896.39111328125,
                404.1787109375
            ]
        }
    ],
    }

    #evaluate(pred, gt,0.7,0.3,True)
    evaluate(pred, gt,0.7,0.3)

        

