# 3Dof-Object-Pose-Estimation
3 Dof object pose estimation with new representation

## Train

python train.py --mode [reg,cls,sord] --net [resnet,shufflenet,mobilenet]

--num_classes [33,66] --num_epochs --lr --lr_decay --unfreeze 

--train_data --valid_data --input_size [224,196,160,128,96] 

--width_mult [1.0,0.5] --batch_size --top_k --cls2reg --alpha

--save_dir
