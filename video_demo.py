import cv2
from test_on_img import Test
from utils import remove_distortion
from net import MobileNetV2
from torchvision import transforms
from net import MobileNetV2
import argparse
import matplotlib.pyplot as plt

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Video Testing.')
    parser.add_argument('--video', dest='video', help='Directory path for data.',
                        default='./videos/test1.mp4', type=str)
    parser.add_argument('--snapshot1', dest='snapshot1', help='Directory path for data.',
                        default='./results/MobileNetV2_1.0_classes_66_input_224/snapshot/MobileNetV2_1.0_classes_66_input_224_epoch_50_front_vector.pkl', type=str)

    parser.add_argument('--snapshot2', dest='snapshot2', help='Directory path for data.',
                        default='./results/MobileNetV2_1.0_classes_66_input_224/snapshot/MobileNetV2_1.0_classes_66_input_224_epoch_17_right_vector.pkl', type=str)

    

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    _, frame = cap.read()

    test = Test(MobileNetV2, MobileNetV2, args.snapshot1, args.snapshot2, 66)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # make 3D axis
    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    while 1:
        _, frame = cap.read()
        draw_img = frame.copy()
	
	frame = remove_distortion(frame)
        img = cv2.resize(frame,(224,224))
        img = transform(img)
        img = img.unsqueeze(0)
	
        ax  = plt.axes(projection='3d')	
        test.test_per_img(img,draw_img,ax)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
              break
 
    cv2.destroyAllWindows()
    cap.release()
	
	
