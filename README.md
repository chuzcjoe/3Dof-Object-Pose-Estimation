# 3Dof-Object-Pose-Estimation
3 Dof object pose estimation with new representation

## Train

python train.py  --num_classes [33,66] --num_epochs --lr --lr_decay --unfreeze 

--train_data --valid_data --input_size [224,196,160,128,96] 

--width_mult [1.0,0.5] --batch_size --top_k --cls2reg --alpha

--save_dir

## Test on testing dataset

python test.py --snapshot --analysis

## Single image testing

python test_on_img.py --img --snapshot

## Video testing

python video_demo.py --video --snapshot


## Dataset
Training size: 7800 <br>
Validation size: 1950<br>

raw video: 1280 * 720    40FPS
processed image(remove distortion): 960 * 720

network input: 244 * 244

## Results (one front-vector)
Avg angle error on training data(50 epoches): 1.779 degree(s)<br>
Avg angle error on validation data(50 epoches): 2.359545946121216 degree(s)<br>
Training loss(50 epoches): x_loss: 0.074635 | y_loss: 0.073857 | z_loss: 0.077535<br>

## Results (one right-vector)
Avg angle error on training data(50 epoches): 2.561 degree(s)<br>
Avg angle error on validation data(50 epoches): 2.7696986198425293 degree(s)<br>
Training loss(50 epoches): x_loss: 0.076573 | y_loss: 0.076306 | z_loss: 0.077133 <br>

## Error distribution on the entire dataset
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/MobileNetV2_1.png" width="800">

## collection score plot (one front-vector)
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/collect_score.png" width="500">

## collection score plot (one right-vector)
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/collect_score_right.png" width="500">

## Video demo

### two vector
[video link(two vector with the tracker)](https://www.youtube.com/watch?v=vHMiGsI2XKM)

[video link(three vector without the tracker)](https://www.youtube.com/watch?v=MIEMjuBjgNg)

## Visualizing predicted right vector
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/merge.jpg" width="600">

## Visualizing predicted right vector + ground-truth labels
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/merge_right+label.jpg" width="600">

## Visualizing predicted front vector
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/merge_front.jpg" width="600">

## Visualizing predicted front vector + ground-truth labels
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/merge_front+label.jpg" width="600">

## Bad Cases

### angle error greater than 10 degrees (front vector)
[bad case images](https://drive.google.com/open?id=1T75OLTHsl9N-bFtu5O1rLZ7FZXL6K0Aa)

### angle error greater than 10 degrees (right vector)
[bad case images](https://drive.google.com/open?id=15oI9Ql7HiOF42i8lP1Yz34YxR2ry7F8E)

