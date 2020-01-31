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

    def __init__(self,model,snapshot,num_classes):
        
        self.num_classes = num_classes
        self.model = model(num_classes=self.num_classes)
        self.saved_state_dict = torch.load(snapshot)
        self.model.load_state_dict(self.saved_state_dict)
        self.model.cuda(0)

        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        self.softmax = nn.Softmax(dim=1).cuda(0)

    def draw_attention_vector(self, pred_vector, img):
        #save_dir = os.path.join(args.save_dir, 'show_front')
        #img_name = os.path.basename(img_path)

        #img = cv.imread(img_path)

        predx, predy, predz = pred_vector
        #img = np.squeeze(img,axis=0)
        #print(img.shape)
        #img = img.reshape((224,224,3))
        

        # draw pred attention vector with red
        utils.draw_front(img, predy, predz, tdx=None, tdy=None, size=100, color=(0, 0, 255))

        #cv.imwrite(os.path.join(save_dir, img_name), img)
        cv.imshow("test_result",img)
        #plt.imshow(img)
        #plt.show()

    def test_per_img(self,cv_img,draw_img):
        with torch.no_grad():
            images = cv_img.cuda(0)
            #print(images.shape)

            # get x,y,z cls predictions
            x_cls_pred, y_cls_pred, z_cls_pred = self.model(images)

            # get prediction vector(get continue value from classify result)
            _, _, _, pred_vector = utils.classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, self.softmax, self.num_classes)

            self.draw_attention_vector(pred_vector[0].cpu().tolist(),
                                          draw_img)

#input_size = 224
#test = Test(MobileNetV2,"./results/MobileNetV2_1.0_classes_66_input_224/snapshot/MobileNetV2_1.0_classes_66_input_224_epoch_50.pkl",66)
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#img = cv.imread("./test_imgs/img_006244.jpg")
#draw_img = img.copy()
#img = cv.resize(img,(224,224))
#img = transform(img)
#img = img.unsqueeze(0)

#test.test_per_img(img,draw_img)

