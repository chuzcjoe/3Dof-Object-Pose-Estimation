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

## Results (one front-vector)
Avg angle error on training data(50 epoches): 1.779 degree(s)<br>
Avg angle error on validation data(50 epoches): 2.359545946121216 degree(s)<br>
Training loss(50 epoches): x_loss: 0.074635 | y_loss: 0.073857 | z_loss: 0.077535<br>

## Results (one right-vector)
Avg angle error on training data(50 epoches): 2.561 degree(s)<br>
Avg angle error on validation data(50 epoches): 2.7696986198425293 degree(s)<br>
Training loss(50 epoches): x_loss: 0.076573 | y_loss: 0.076306 | z_loss: 0.077133 <br>

## collection score plot (one front-vector)
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/collect_score.png" width="500">


## Video demo link (one front-vector)

[youtube(one vector)](https://www.youtube.com/watch?v=Gxo8jXZ0b2Q)

## Visualizing predicted right vector
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/merge.jpg" width="800">

## Visualizing predicted right vector + ground-truth lables
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/merge_right+label.jpg" width="800">

## Visualizing predicted front vector
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/merge_front.jpg" width="800">

## Visualizing predicted front vector + ground-truth lables
<img src="https://github.com/chuzcjoe/3Dof-Object-Pose-Estimation/raw/master/imgs/merge_front+label.jpg" width="800">

