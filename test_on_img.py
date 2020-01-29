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


class Test():

    def __int__(self,model,snapshot,num_classes=66):
        self.model = model(num_classes=num_classes)

        self.saved_state_dict = torch.load(snapshot)
        self.model.load_state_dict(self.saved_state_dict)
        self.model.cuda(0)

        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        self.softmax = nn.Softmax(dim=1).cuda(0)

    def draw_attention_vector(self, pred_vector, cv_img):
        #save_dir = os.path.join(args.save_dir, 'show_front')
        #img_name = os.path.basename(img_path)

        #img = cv.imread(img_path)

        predx, predy, predz = pred_vector

        # draw pred attention vector with red
        utils.draw_front(cv_img, predy, predz, tdx=None, tdy=None, size=100, color=(0, 0, 255))

        #cv.imwrite(os.path.join(save_dir, img_name), img)

    def test_per_img(self,cv_img,num_classes=66):
        with torch.no_grad():
            images = cv_img.cuda(0)

            # get x,y,z cls predictions
            x_cls_pred, y_cls_pred, z_cls_pred = model(images)

            # get prediction vector(get continue value from classify result)
            _, _, _, pred_vector = utils.classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, self.softmax, num_classes)

            draw_attention_vector(pred_vector.cpu().tolist(),
                                          cv_img)







