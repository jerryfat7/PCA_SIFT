# READ ME

## 环境配置

目标检测模型采用：openmmlab mmdetection

### 创建环境

conda create --name openmmlab python=3.8 -y

conda activate openmmlab

 

### 安装pytorch

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

 

### 安装MMEngine和MMCV

pip install -U openmim

mim install mmengine

mim install "mmcv>=2.0.0"

 

### 安装mmdetection

mim install mmdet

 

需要复制checkpoints和config文件夹到代码目录，里面是目标检测的模型定义和模型数据，链接详见附录



### 安装opencv

pip install opencv-python

opencv的补充，需要调用opencv中实现的SIFT算法

pip install opencv-contrib-python

 

### 安装sklearn

pip install scikit-learn



### optional

若需要进行标注/可视化标注结果，需要安装pyqt5

pip install pyqt5



## 各模块作用

### myDataSet.py

提供对数据集的分割、判断正负样本、读取图片、获得标注数据等

### SIFT_matcher.py

基于SIFT模型的匹配算法实现，还提供包括SIFT的图片归一化方法

### PCA_matcher.py

基于PCA模型的匹配算法实现，还提供包括PCA的图片归一化方法

### SIFT_main.py

运行SIFT算法的程序入口

### PCA_main.py

运行PCA算法的程序入口

### majority_voting.py

ensemble方法的程序入口

### evaluater.py

提供对一个场景和一个目标杯子进行检测后的评价方法

### detecter.py

目标检测模型的封装类，调用mmdet库实现

### boxes.py

检测框相关的方法封装，参考开源代码实现

### classificationBoundary.py

基于SVM的分类面学习器，可以调用其中的函数也可以在该文件中指定运行结果目录生成分类面

### labeler_main.py 和 labeler_ui.py

标注器的主函数和ui编译后的.py文件，注意运行main后会重定义方向键

### mylogging.py

用于记录训练过程和结果

### det_utils.py

参考开源代码，但仅使用其中的iou计算函数



## 附录

### 1.checkpoints和configs文件夹的下载链接

configs_chekpoints

下载解压后将解压的两个文件夹放置于与代码在同一目录下

### 2.标注后的数据集下载

Scenes.zip

仅标注了Scene的相关文件，故不提供mugs的数据，解压后替换Scene文件夹即可

### 3.预识别方框信息

pre_detect_info.zip

由于多次重复实验的目标结果一致，故默认生成一次后存到磁盘，再次运行到相同的场景

图时直接读取磁盘数据而不是再次推理浪费时间，该文件解压到与代码同一目录下

可选，若不下载会自动重新推理并生成这几个文件



https://cloud.tsinghua.edu.cn/d/7f6729245fe241499b48/















