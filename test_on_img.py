# -*- coding:utf-8 -*-
import os
import tqdm
import utils
import torch
import argparse
import cv2 as cv
import numpy as np
import torch.nn as nn
from net import MobileNetV2
from dataset import loadData
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
import matplotlib.pyplot as plt

class Test:

    def __init__(self,model1,model2,snapshot1,snapshot2,num_classes):
        
        self.num_classes = num_classes
        self.model1 = model1(num_classes=self.num_classes)
        self.model2 = model2(num_classes=self.num_classes)

        self.saved_state_dict1 = torch.load(snapshot1)
        self.model1.load_state_dict(self.saved_state_dict1)
        self.model1.cuda(0)

        self.saved_state_dict2 = torch.load(snapshot2)
        self.model2.load_state_dict(self.saved_state_dict2)
        self.model2.cuda(0)


        self.model1.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        self.model2.eval()

        self.softmax = nn.Softmax(dim=1).cuda(0)

    def draw_attention_vector(self, pred_vector1, pred_vector2, img, ax):
        #save_dir = os.path.join(args.save_dir, 'show_front')
        #img_name = os.path.basename(img_path)

        #img = cv.imread(img_path)

        predx, predy, predz = pred_vector1
        #img = np.squeeze(img,axis=0)
        #print(img.shape)
        #img = img.reshape((224,224,3))
        

        # draw pred attention vector with red
        utils.draw_front(img, predy, predz, tdx=None, tdy=None, size=100, color=(0, 0, 255))


        predx, predy, predz = pred_vector2
        utils.draw_front(img, predy, predz, tdx=None, tdy=None, size=100, color=(0, 255, 0))
        
        #cv.imwrite(os.path.join(save_dir, img_name), img)
        cv.imshow("test_result",img)
        utils.draw_3d_coor(pred_vector1, pred_vector2, ax)
        #plt.imshow(img)
        #plt.show()

    def test_per_img(self,cv_img,draw_img, ax):
        with torch.no_grad():
            images = cv_img.cuda(0)
            #print(images.shape)

            # get x,y,z cls predictions
            x_cls_pred, y_cls_pred, z_cls_pred = self.model1(images)

            # get prediction vector(get continue value from classify result)
            _, _, _, pred_vector1 = utils.classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, self.softmax, self.num_classes)



            # get x,y,z cls predictions
            x_cls_pred, y_cls_pred, z_cls_pred = self.model2(images)

            # get prediction vector(get continue value from classify result)
            _, _, _, pred_vector2 = utils.classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, self.softmax, self.num_classes)

            self.draw_attention_vector(pred_vector1[0].cpu().tolist(), pred_vector2[0].cpu().tolist(),
                                          draw_img,ax)

#input_size = 224
#test = Test(MobileNetV2,"./results/MobileNetV2_1.0_classes_66_input_224/snapshot/MobileNetV2_1.0_classes_66_input_224_epoch_50.pkl",66)
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#img = cv.imread("./test_imgs/img_006244.jpg")
#draw_img = img.copy()
#img = cv.resize(img,(224,224))
#img = transform(img)
#img = img.unsqueeze(0)

#test.test_per_img(img,draw_img)

