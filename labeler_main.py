from labeler_ui import Ui_labeler
import os,sys
from PyQt5.QtWidgets import QMessageBox, QWidget, QApplication
from PyQt5 import QtCore,QtGui
from PyQt5.QtGui import QPixmap,QImage,QDoubleValidator
from win32ui import CreateFileDialog
import json
#from PyQt5.QtCore import QTimer
#from threading import Thread
#import time
import traceback
from PIL import Image,ImageDraw,ImageFont
from detecter import detecter

fontsize = 15
font = ImageFont.truetype("arial.ttf", fontsize)

default_threshold = 0.2

class labeler_main(QWidget):
    def __init__(self):
        super(labeler_main, self).__init__()
        self.ui = Ui_labeler()
        self.detecter = detecter()

        self.targetItem = "cup"

        self.pictures = []
        self.thisPic = None
        self.curdir = None
        self.fileIter = None
        #[xmin,ymin,xmax,ymax]
        self.bboxes = []

        self.ui.setupUi(self)
        self.lineEdits = []
        self.lineEdits.append(self.ui.lineEdit_1)
        self.lineEdits.append(self.ui.lineEdit_2)
        self.lineEdits.append(self.ui.lineEdit_3)
        self.lineEdits.append(self.ui.lineEdit_4)
        self.lineEdits.append(self.ui.lineEdit_5)
        self.lineEdits.append(self.ui.lineEdit_6)
        self.lineEdits.append(self.ui.lineEdit_7)
        self.lineEdits.append(self.ui.lineEdit_8)
        self.lineEdits.append(self.ui.lineEdit_9)
        self.maxBoxes = len(self.lineEdits)

        self.ui.pushButton_fileSelect.clicked.connect(self.selectFile)
        self.ui.pushButton_filePrev.clicked.connect(self.prevFile)
        self.ui.pushButton_fileNext.clicked.connect(self.nextFile)
        self.ui.pushButton_SaveRes.clicked.connect(self.saveRes)
        self.ui.pushButton_redetect.clicked.connect(self.redetect)
        self.ui.pushButton_setDefalut.clicked.connect(self.setDefaultThreshold)
        #仅能输入0-1之间的浮点数
        self.ui.lineEdit_threshold.setValidator(QDoubleValidator(0.01,0.99,3))
        self.ui.lineEdit_threshold.setText(str(default_threshold))

        #模型选择，默认模型和fast rcnn
        self.ui.comboBox_modelChoose.addItems(["default","faster r cnn"])
        faster_r_cnn_config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
        faster_r_cnn_checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.modelInfo = [None,[faster_r_cnn_config_file,faster_r_cnn_checkpoint_file]]
        self.nowModelId = 0

        #模型选择变更绑定
        self.ui.comboBox_modelChoose.activated.connect(self.changeModel)
        #combobox需要屏蔽key相关事件
        self.ui.comboBox_modelChoose.installEventFilter(self)

        self.init = False

    def eventFilter(self, watched, event):
        if watched == self.ui.comboBox_modelChoose and\
        (event == QtCore.QEvent.Type.KeyPress or\
          event == QtCore.QEvent.Type.KeyRelease):
            #过滤
            return True
        
            #return super.eventFilter(self, watched, event)
        return False

    def keyPressEvent(self, a0: QtGui.QKeyEvent):
        return

    #上：保存；左：上一张图；右：下一张图；下：重新检测；ctrl：更换模型
    def keyReleaseEvent(self, a0: QtGui.QKeyEvent):
        if not self.init:
            return
        if a0.key() == QtCore.Qt.Key_Left:
            self.prevFile()
        elif a0.key() == QtCore.Qt.Key_Right:
            self.nextFile()
        elif a0.key() == QtCore.Qt.Key_Up:
            #print("Key_Space")
            self.saveRes()
        elif a0.key() == QtCore.Qt.Key_Control:
            #print("control")
            id = self.ui.comboBox_modelChoose.currentIndex()
            newId = (id+1) % self.ui.comboBox_modelChoose.count()
            self.ui.comboBox_modelChoose.setCurrentIndex(newId)
            self.changeModel()
        elif a0.key() == QtCore.Qt.Key_Down:
            try:
                self.redetect()
            except Exception as e:
                traceback.print_exc()
        
        
    def selectFile(self):
        try:
            #fspec = "JPEG Files (*.jpeg or *.jpg)|*.jpeg;*.jpg|PNG Files (*.png)|*.png|BMP Files (*.bmp)|*.bmp|"
            fspec = "PNG Files (*.png)|*.png|JPEG Files (*.jpeg or *.jpg)|*.jpeg;*.jpg|BMP Files (*.bmp)|*.bmp|"
            dlg = CreateFileDialog(1, None , None, 1, fspec, None)
            cwd = os.getcwd()
            #print(cwd)
            dlg.SetOFNInitialDir(cwd)
            flag = dlg.DoModal()
            # 设置文件扩展名过滤,用双分号间隔
            fileName_choose = dlg.GetPathName()
            #print(fileName_choose)
            #没有选中图片，直接返回
            if not os.path.isfile(fileName_choose):
                return
            self.init = True

            fs = fileName_choose.split("\\")
            #print(fs)
            #初始化当前文件夹信息和当前文件
            self.curdir = fs[0] +'\\' + os.path.join(*fs[1:-1])
            #print(self.curdir)
            #筛选所有图片格式文件
            files = os.listdir(self.curdir)
            self.pictures = []
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or \
                file.endswith(".png") or file.endswith(".bmp"):
                   self.pictures.append(file)  
            
            self.thisPic = fs[-1]
            self.fileIter = self.pictures.index(self.thisPic)
            #设置默认阈值
            self.ui.lineEdit_threshold.setText(str(default_threshold))
            self.update()

        except Exception as e:
            traceback.print_exc()
        return


    def update(self):
        if not self.init:
            return

        fileFullPath = os.path.join(self.curdir,self.thisPic)
        #label更新
        self.ui.label_filePath.setText(fileFullPath)
        
           
        #若已有标注结果，在图片上显示红色框框
        #解析json获取已标注信息
        allData = self.loadPicLabel()
        if (not allData is None) and self.thisPic in allData.keys():
            self.bboxes = []
            bboxes_num = 0
            PIL_img = Image.open(fileFullPath)
            draw = ImageDraw.Draw(PIL_img)
            t_list = allData[self.thisPic]
            for dataForEachBox in t_list:
                if dataForEachBox["label"] != "":
                    bboxes_num += 1
                    draw.rectangle(dataForEachBox["bbox"],outline="red")
                    draw.text([dataForEachBox["bbox"][0]+3,dataForEachBox["bbox"][1]],"label:"+dataForEachBox["label"],fill ="red",font=font)
                    self.bboxes.append(dataForEachBox["bbox"])
            qpixmap = pil2pixmap(PIL_img)
            self.showImg(qpixmap)
            #只允许识别到的输入框
            self.setLineEditCanWrite(bboxes_num)
            # t_list = allData[self.thisPic]
            # for i in range(len(t_list)):
            #     self.lineEdits[i].setText(t_list[i]["label"])

            return
        
        #否则自动检测bbox,并显示白色框框
        else:
            self.redetect()

        
        #清空所有的输入
        #self.clearAllLineEdit()
        
        return
    
    def redetect(self):
        fileFullPath = os.path.join(self.curdir,self.thisPic)
        threshold = float(self.ui.lineEdit_threshold.text())
        bboxes = self.detecter.detect(fileFullPath, self.targetItem,threshold=threshold).tolist()
        if len(bboxes) > self.maxBoxes:
            bboxes = bboxes[0:self.maxBoxes]
        self.bboxes = bboxes
        #print(bboxes)
        bboxes_num = len(bboxes)
        #锁定部分输入框
        self.setLineEditCanWrite(bboxes_num)
        #show img
        PIL_img = Image.open(fileFullPath)
        draw = ImageDraw.Draw(PIL_img)
        for i in range(bboxes_num):
            draw.rectangle(bboxes[i])
            draw.text([bboxes[i][0]+3,bboxes[i][1]],"bbox:{}".format(i+1),font=font)
        
        qpixmap = pil2pixmap(PIL_img)
        self.showImg(qpixmap)
        
        return
    # def update(self):
    #     if not self.init:
    #         return

    #     fileFullPath = os.path.join(self.curdir,self.thisPic)
    #     #label更新
    #     self.ui.label_filePath.setText(fileFullPath)
    #     #showImg
    #     threshold = float(self.ui.lineEdit_threshold.text())
    #     bboxes = self.detecter.detect(fileFullPath, self.targetItem,threshold=threshold)
    #     if len(bboxes) > 5:
    #         bboxes = bboxes[0:5]
    #     self.bboxes = bboxes
    #     #print(bboxes)
    #     PIL_img = Image.open(fileFullPath)
    #     draw = ImageDraw.Draw(PIL_img)
    #     bboxes_num = len(bboxes)
    #     for i in range(bboxes_num):
    #         draw.rectangle(bboxes[i])
    #         draw.text(bboxes[i][0:2],"bbox:{}".format(i+1))
        
    #     qpixmap = pil2pixmap(PIL_img)
    #     self.showImg(qpixmap)


    #     #锁定部分输入框
    #     self.setLineEditCanWrite(bboxes_num)
    #     #Todo:解析json获取已标注信息
        
    #     #先清空所有的输入
    #     #self.clearAllLineEdit()
        
    #     allData = self.loadPicLabel()
    #     if (not allData is None) and self.thisPic in allData.keys():
    #         t_list = allData[self.thisPic]
    #         for i in range(len(t_list)):
    #             self.lineEdits[i].setText(t_list[i]["label"])

    #     return

    def nextFile(self):
        if not self.init:
            return
        if  self.fileIter is None or (self.fileIter + 1) == len(self.pictures):
            return
        self.fileIter += 1
        self.thisPic = self.pictures[self.fileIter]
        self.ui.lineEdit_threshold.setText(str(default_threshold))
        self.update()

        return

    def prevFile(self):
        if not self.init:
            return
        if self.fileIter is None or self.fileIter == 0:
            return

        self.fileIter -= 1
        self.thisPic = self.pictures[self.fileIter]
        self.ui.lineEdit_threshold.setText(str(default_threshold))
        self.update()

        return

    def saveRes(self):
        if not self.init:
            return
        jsonPath = os.path.join(self.curdir,"annotation.json")
        allData = self.loadPicLabel()
        if allData is None:
            allData = dict()
        
        allData[self.thisPic] = []
        for i in range(len(self.bboxes)):
            label = self.lineEdits[i].text()
            if label == "":
                continue
            t_dict = dict()
            t_dict["label"] = label
            t_dict["bbox"] = self.bboxes[i]
            allData[self.thisPic].append(t_dict)
        
        data_write = json.dumps(allData,indent=4)
        with open(jsonPath, 'w') as f_json:
            f_json.write(data_write)

        self.update()
        return

    def loadPicLabel(self):
        jsonPath = os.path.join(self.curdir,"annotation.json")
        if not os.path.exists(jsonPath):
            return None
        else:
            with open(jsonPath) as json_file:
                json_data = json.load(json_file)
            return json_data

    def showImg(self,Qpixmap_img:QPixmap):
        self.ui.label_img.setPixmap(Qpixmap_img)
        self.ui.label_img.setScaledContents(True)
        return
    
    def setLineEditCanWrite(self,bboxnum):
        assert  (bboxnum >= 0 and bboxnum <= self.maxBoxes)
        for i in range(len(self.lineEdits)):
            if i < bboxnum:
                self.lineEdits[i].setReadOnly(False)
            else:
                self.lineEdits[i].setReadOnly(True)
                self.lineEdits[i].setText("")

    def clearAllLineEdit(self):
        for i in range(len(self.lineEdits)):
            self.lineEdits[i].setText("")
        return

    def setDefaultThreshold(self):
        threshold = float(self.ui.lineEdit_threshold.text())
        global default_threshold
        default_threshold = threshold
        return

    def changeModel(self):
        newModelId = self.ui.comboBox_modelChoose.currentIndex()
        
        if self.nowModelId == newModelId:
            return
        else:
            self.nowModelId = newModelId
            if self.modelInfo[newModelId] is None:
                self.detecter = detecter()
            else:
                configfile = self.modelInfo[newModelId][0]
                checkpoint = self.modelInfo[newModelId][1]
                self.detecter = detecter(configfile, checkpoint)
            return


def pil2pixmap(im):

    if im.mode == "RGB":
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
    elif  im.mode == "RGBA":
        r, g, b, a = im.split()
        im = Image.merge("RGBA", (b, g, r, a))
    elif im.mode == "L":
        im = im.convert("RGBA")
    # Bild in RGBA konvertieren, falls nicht bereits passiert
    im2 = im.convert("RGBA")
    data = im2.tobytes("raw", "RGBA")
    qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
    pixmap = QPixmap.fromImage(qim)
    return pixmap

if __name__ == '__main__':


    app = QApplication(sys.argv)  
    #using faster rcnn
    # config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # myMainWindow = labeler_main([config_file,checkpoint_file])
    myMainWindow = labeler_main()
    #myMainWindow.show()
    myMainWindow.activateWindow()
    myMainWindow.setWindowState(QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive )
    myMainWindow.showNormal()
    sys.exit(app.exec_())