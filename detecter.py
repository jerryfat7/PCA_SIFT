import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.visualization import Visualizer
import numpy as np
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 指定模型的配置文件和 checkpoint 文件路径
# rtm模型
default_config_file = 'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
default_checkpoint_file = 'checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
#faster r cnn的配置文件和 checkpoint 文件路径
faster_r_cnn_config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
faster_r_cnn_checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

class detecter():
    def __init__(self, config_file = default_config_file, checkpoint_file = default_checkpoint_file):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = init_detector(config_file, checkpoint_file, device=device)

    # return a numpy array
    # last dim is [xmin,ymin,xmax,ymax]
    def detect(self, img, itemName, threshold=0.5):
        result = inference_detector(self.model, img)
        try:
            item_label_id =  list(self.model.dataset_meta['classes']).index(itemName)
        except Exception as e:
            print("label中没有这样的物体:{}".format(itemName))
            raise e
        bboxes = result.pred_instances.bboxes.cpu()
        labels = result.pred_instances.labels.cpu()
        scores = np.array(result.pred_instances.scores.cpu())
        inds_above_threshold = np.where(scores>threshold)
        inds_item = np.where(labels == item_label_id)
        inds = np.intersect1d(inds_above_threshold,inds_item)

        #按xmin左到右的顺序排列
        bboxes_np = np.array(bboxes[inds])
        t_arg = bboxes_np.argsort(axis=0)
        bboxes_np = bboxes_np[t_arg[:,0],:]

        return bboxes_np

    def detect_sortedByScore(self, img, itemName):
        result = inference_detector(self.model, img)
        try:
            item_label_id =  list(self.model.dataset_meta['classes']).index(itemName)
        except Exception as e:
            print("label中没有这样的物体:{}".format(itemName))
            raise e
        bboxes = result.pred_instances.bboxes.cpu().numpy().reshape(-1,4)
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        inds_item = np.where(labels == item_label_id)

        bboxes_item = bboxes[inds_item]
        scores_item = scores[inds_item]
        #评分从大到小排列
        inds = (-scores_item).argsort()
        return bboxes_item[inds],scores_item[inds]
    
    # 过滤detect结果，只保留大于detectThreshold和数量小于等于maxCandidateItems的结果
    # maxCandidateItems用于限制detecter检测到的物体数量
    # score小于detectThreshold的物体会被过滤
    def filtWithScores(bboxes,scores,detectThreshold, maxCandidateItems=10):
        indsAboveThres = np.where(scores>=detectThreshold)
        bboxes = bboxes[indsAboveThres,:]
        scores = scores[indsAboveThres]

        bboxes = bboxes[0:maxCandidateItems,:]
        scores = scores[0:maxCandidateItems]
        return bboxes,scores


if __name__ == "__main__":
    #for test
    import det_utils
    import boxes

    myDetecter_default = detecter()
    myDetecter_fastrcnn = detecter(faster_r_cnn_config_file, faster_r_cnn_checkpoint_file)

    imgPath = 'demo/scene_43_00000.png'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
    #bboxes = myDetecter.detect(imgPath,'cup',0.5)
    t_bboxes,scores = myDetecter_default.detect_sortedByScore(imgPath,'cup')
    print(t_bboxes.shape)
    t_bboxes = t_bboxes[0:7,:]
    bboxes = t_bboxes.tolist()
    scores = scores.tolist()
    bboxes = bboxes[0:7]

    t_bboxes1,scores1 = myDetecter_fastrcnn.detect_sortedByScore(imgPath,'cup')
    print(t_bboxes1.shape)
    t_bboxes1 = t_bboxes1[0:7,:]
    bboxes1 = t_bboxes1.tolist()
    scores1 = scores1.tolist()


    #print(bboxes)
    # 显示结果
    img = mmcv.imread(imgPath)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer = Visualizer(image=img, vis_backends=[dict(type='LocalVisBackend')],
                        save_dir='temp_dir')
    for i in range(len(bboxes)):
        visualizer.draw_bboxes(np.array(bboxes[i])).draw_texts('cup,score:{:.3f}'.format(scores[i]),np.array(bboxes[i][0:2]),font_sizes=20)
    
    for i in range(len(bboxes1)):
        visualizer.draw_bboxes(np.array(bboxes1[i]),edge_colors='r').draw_texts('cup,score:{:.3f}'.format(scores1[i]),np.array(bboxes1[i][0:2]),font_sizes=20,colors='r')
    visualizer.add_image('demo', visualizer.get_image())
    #visualizer.show()
    
    matcher = det_utils.Matcher(high_threshold=0.7, low_threshold=0.3, allow_low_quality_matches=False)
    ious = boxes.box_iou(t_bboxes,t_bboxes1)
    ids_in_t_bboxes = matcher(ious)
    print(ids_in_t_bboxes)
    print(ious.shape)
    visualizer1 = Visualizer(image=img,vis_backends=[dict(type='LocalVisBackend')],
                        save_dir='temp_dir')
    
    for i in range(ids_in_t_bboxes.shape[0]):
        if ids_in_t_bboxes[i] < 0:
            continue
        t_box = t_bboxes[ids_in_t_bboxes[i],:]
        visualizer1.draw_bboxes(np.array(t_box)).draw_texts('cup,{}'.format(i),np.array(t_box[0:2]),font_sizes=20)
    visualizer1.add_image('after ious matches', visualizer1.get_image())
    #visualizer1.show()

