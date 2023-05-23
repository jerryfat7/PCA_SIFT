import os
import os.path as osp
import json
from PIL import Image
import numpy as np
import cv2

defaultPath = r"..\Personalized_Segmentation"
class myDataSet():
    def __init__(self, dataRootPath = defaultPath):
        self.clothMugsSuffixes = ["02","06","07","10","13","14","15","16","17","18","19"]

        self.objPath = osp.join(dataRootPath, "Objects","mugs")
        self.scenePath = osp.join(dataRootPath, "Scenes")
        
        tmp_trainObjSuffix = [1,2,6,8,9,10,12,15,18]
        self.trainObjFolders = []
        for sfx in tmp_trainObjSuffix:
            str_sfx = "{}".format(sfx).zfill(2)
            self.trainObjFolders.append("mug_{}".format(str_sfx))
        
        tmp_ValObjSuffix = [3,4,5,7,11]
        self.ValObjFolders = []
        for sfx in tmp_ValObjSuffix:
            str_sfx = "{}".format(sfx).zfill(2)
            self.ValObjFolders.append("mug_{}".format(str_sfx))

        tmp_testObjSuffix = [13,14,16,17,19]
        self.testObjFolders = []
        for sfx in tmp_testObjSuffix:
            str_sfx = "{}".format(sfx).zfill(2)
            self.testObjFolders.append("mug_{}".format(str_sfx))

        tmp_trainSceneSuffix_3 = [3,4,5,7,24,30,9,10,20,22]
        tmp_trainSceneSuffix_5 = [38,39,40,41,42,45]
        self.trainSceneFolders_3 = []
        self.trainSceneFolders_5 = []
        for sfx in tmp_trainSceneSuffix_3:
            str_sfx = "{}".format(sfx).zfill(2)
            self.trainSceneFolders_3.append("scene_{}".format(str_sfx))
        for sfx in tmp_trainSceneSuffix_5:
            str_sfx = "{}".format(sfx).zfill(2)
            self.trainSceneFolders_5.append("scene_{}".format(str_sfx))
        self.trainSceneFolders = self.trainSceneFolders_3 + self.trainSceneFolders_5


        tmp_ValSceneSuffix_3 = [1,2,6,25]
        tmp_ValSceneSuffix_5 = [46,47]
        self.ValSceneFolders_3 = []
        self.ValSceneFolders_5 = []
        for sfx in tmp_ValSceneSuffix_3:
            str_sfx = "{}".format(sfx).zfill(2)
            self.ValSceneFolders_3.append("scene_{}".format(str_sfx))
        for sfx in tmp_ValSceneSuffix_5:
            str_sfx = "{}".format(sfx).zfill(2)
            self.ValSceneFolders_5.append("scene_{}".format(str_sfx))
        self.ValSceneFolders = self.ValSceneFolders_3 + self.ValSceneFolders_5


        tmp_testSceneSuffix_3 = [23,27]
        tmp_testSceneSuffix_5 = [43]
        self.testSceneFolders_3 = []
        self.testSceneFolders_5 = []
        for sfx in tmp_testSceneSuffix_3:
            str_sfx = "{}".format(sfx).zfill(2)
            self.testSceneFolders_3.append("scene_{}".format(str_sfx))
        for sfx in tmp_testSceneSuffix_5:
            str_sfx = "{}".format(sfx).zfill(2)
            self.testSceneFolders_5.append("scene_{}".format(str_sfx))
        self.testSceneFolders = self.testSceneFolders_3 + self.testSceneFolders_5

    def getDataSet(self, mode:str = "train"):
        mode = mode.lower()
        if mode == "train":
            return self.trainObjFolders,self.trainSceneFolders_3,self.trainSceneFolders_5
        elif mode == "val" or mode == "validation":
            return self.ValObjFolders,self.ValSceneFolders_3,self.ValSceneFolders_5
        elif mode == "test":
            return self.testObjFolders,self.testSceneFolders_3,self.testSceneFolders_5
        else:
            raise NotImplementedError


    def GetPictures(self, folder:str):
        if folder.startswith("mug"):
            folderPath = osp.join(self.objPath, folder)
        elif folder.startswith("scene"):
            folderPath = osp.join(self.scenePath, folder)
        else:
            raise NotImplementedError
        
        mugsWithCloth = []
        if folder.startswith("mug"):
            for suffix in self.clothMugsSuffixes:
                if folder.endswith(suffix):
                    folderClothPath = osp.join(self.objPath, folder+"_cloth")
                    mugsWithCloth = os.listdir(folderClothPath)
                    break


        files = os.listdir(folderPath)
        pictures = []
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or \
            file.endswith(".png") or file.endswith(".bmp"):
                pictures.append(file)  

        return pictures,mugsWithCloth
    

    def ReadMugsPictures(self, mugsFolder:str, normalize = None):
        assert mugsFolder.startswith("mug")
        if normalize == None:
            normalize = lambda x:x
        objs = []
        objFileNames,objFileNames_cloth = self.GetPictures(mugsFolder)
        #先不考虑带cloth的
        for objFileName in objFileNames:
            imgPath = osp.join(self.objPath, mugsFolder, objFileName)
            imgPIL = Image.open(imgPath)
            objs.append(normalize(np.asarray(imgPIL)))
        return objs
    
    def ReadScenePictures(self, SceneFolder:str, normalize = None):
        assert SceneFolder.startswith("scene")
        if normalize == None:
            normalize = lambda x:x
        scenesImgs = []
        scenesFileNames,_ = self.GetPictures(SceneFolder)
        
        for sceneFileName in scenesFileNames:
            imgPath = osp.join(self.scenePath, SceneFolder, sceneFileName)
            imgPIL = Image.open(imgPath)
            scenesImgs.append(normalize(np.asarray(imgPIL)))

        return scenesImgs
    
    #从根据标注数据得到杯子是否在场景中
    #若不在说明为负样本
    def isPositiveSample(self, mugsFolder:str, SceneFolder):
        assert mugsFolder.startswith("mug")
        assert SceneFolder.startswith("scene")

        jsonPath = osp.join(self.scenePath, "cups_in_scene.json")
        with open(jsonPath) as json_file:
            cups_in_scene = json.load(json_file)

        cups = cups_in_scene[SceneFolder]

        Label = str(int(mugsFolder[-2:]))
        #print(Label)
        #print(cups_in_scene)
        return Label in cups



#     def ReadPictures(self, mugsFolder:str, SceneFolder:str, normalize = None):
#         assert mugsFolder.startswith("mug")
#         assert SceneFolder.startswith("scene")
#         if normalize == None:
#             normalize = lambda x:x
#         objs = []
#         scenesImgs = []

#         objFileNames,objFileNames_cloth = self.GetPictures(mugsFolder)
#         scenesFileNames,_ = self.GetPictures(SceneFolder)
        
        
#         #先不考虑带cloth的
#         for objFileName in objFileNames:
#             imgPath = osp.join(self.objPath, mugsFolder, objFileName)
#             imgPIL = Image.open(imgPath)
#             objs.append(normalize(np.asarray(imgPIL)))

#         for sceneFileName in scenesFileNames:
#             imgPath = osp.join(self.scenePath, SceneFolder, sceneFileName)
#             imgPIL = Image.open(imgPath)
#             scenesImgs.append(normalize(np.asarray(imgPIL)))

#         return objs, scenesImgs
#         #return objFileNames, objFileNames_cloth, ScenesFileNames

    def GetAnnotation(self, SceneFolder:str):
        assert SceneFolder.startswith("scene")

        jsonPath = osp.join(self.scenePath, SceneFolder, "annotation.json")
        with open(jsonPath) as json_file:
            annoData = json.load(json_file)

        return annoData

if __name__ == "__main__":
    md =  myDataSet()
    Mugs, Scenes3, Scenes5= md.getDataSet("train")
    print(Mugs)
    print(Scenes3)
    #print(md.GetPictures(Mugs[0]))


    print(md.isPositiveSample(Mugs[0], Scenes3[0]))
    #print(md.ReadPictures(testMugs[0],testScenes3[0]))