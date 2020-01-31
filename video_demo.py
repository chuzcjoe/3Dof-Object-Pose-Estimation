import cv2
from test_on_img import Test
from net import MobileNetV2
from torchvision import transforms
from net import MobileNetV2
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Video Testing.')
    parser.add_argument('--video', dest='video', help='Directory path for data.',
                        default='./videos/test1.mp4', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Directory path for data.',
                        default='./results/MobileNetV2_1.0_classes_66_input_224/snapshot/MobileNetV2_1.0_classes_66_input_224_epoch_50.pkl', type=str)

    

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    _, frame = cap.read()

    test = Test(MobileNetV2, args.snapshot, 66)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



    while 1:
        _, frame = cap.read()
        draw_img = frame.copy()

        img = cv2.resize(frame,(224,224))
        img = transform(img)
        img = img.unsqueeze(0)

        test.test_per_img(img,draw_img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
              break
 
    cv2.destroyAllWindows()
    cap.release()
	
	
