# FR-AGCN
Forward-reverse Adaptive Graph Convolutional Network for Skeleton-Based Action Recognition

# Abstract
In this work, we propose the novel forward-reverse adaptive graph convolutional networks (FR-AGCN) for skeleton-based action recognition. The sequences of joints and bones, as well as their reverse information, are modeled in the multi-stream networks at the same time. By extracting the features of forward and reverse deep information and performing multi-stream fusion, this strategy can significantly improve the recognition accuracy.

# Environment

 PyTorch version  >=0.4

# Notes

 We separate the three datasets for experiments: 'NTU RGB+D' \& 'NTU RGB+D 120' \& UAV-Human.
 
 Here are some important notes:
 
 1. Please perform data preprocessing before training.

 We set two parameters when using the interframe interpolation strategy for data augmentation, i.e., **fu** and **S**. The frame numbers of all samples are unified to fu first. Here we define the data segmentation factor S, which means that down-sampling is performed every S frames in the temporal dimension.
 
 You can try to change these parameters to increase or decrease the size of the input data, which will have a certain impact on the final model performance.
 
 If the memory is not enough, it is recommended to separate the benchmark for preprocessing. Moreover, it is recommended to set enough virtual memory.
 
 We conducted detailed experiments on the CS benchmark of NTU 60. The results are as follows:
 
| fu | S  | FJ(%) | FB(%) | FJB(%) |
| -- | -- | ----  | ----  | ------ |
|600 | 1  | 86.85 | 86.81 | 88.88  |
|**600** | **2**  | **87.44** | **87.68** | **89.29**  |
|600 | 3  | 86.02 | 87.06 | 88.77  |
|600 | 4  | 85.93 | 86.69 | 88.34  |
|300 | 1  | 86.42 | 86.79 | 88.86  |
|300 | 2  | 85.73 | 85.98 | 88.23  |

Therefore, it is recommended to set fu=600 and S=2 to process three datasets. If the GPU memory is insufficient, please reduce the batchsize as appropriate during training.

 2. Please use the parameters saved in the training process to test before performing the multi-stream fusion operation.

You need to select the parameter file based on the training result and modify the test file in the config.

 3. For ease of description, we define single-stream and multi-stream networks according to input data. Specifically, for single-stream input, FJ-AGCN, RJ-AGCN, FB-AGCN, RB-AGCN indicate that the input to the AGCN are forward joints data, reverse joints data, forward bones data, reverse bones data, respectively. For multi-stream input, FR-AGCN represents the networks that integrates the above four single streams. Moreover, FJB-AGCN means that FJ-AGCN and FB-AGCN are terminally fused, FRJ-AGCN indicates that FJ-AGCN and RJ-AGCN are finally fused. RJB-AGCN, FRB-AGCN can be deduced by analogy.
 
Here, we compare the performance of using each type of input data separately and perform score fusion to obtain the final prediction. The results based on AGCN are shown as follows:
 
|Methods | CS(%) | CV(%) |X-Sub(%)|X-Set(%)| CSv1(%)| CSv2(%)|
| ------ | ----- | ----  | -----  | ------ | ------ | ------ |
|FJ-AGCN | 87.44 | 94.08 | 81.23  | 81.57  | 40.08  | 65.66  |
|RJ-AGCN | 87.78 | 94.00 | 81.23  | 82.14  | 39.23  | 63.40  |
|FB-AGCN | 87.68 | 93.98 | 83.52  | 83.64  | 38.43  | 63.15  |
|RB-AGCN | 88.03 | 93.66 | 83.44  | 83.66  | 38.86  | 63.75  |
|FRJ-AGCN| 88.74 | 95.17 | 83.25  | 83.86  | 41.97  | 67.68  |
|FRB-AGCN| 89.55 | 94.99 | 85.62  | 85.50  | 41.13  | 66.51  |
|FJB-AGCN| 89.29 | 95.34 | 85.58  | 85.77  | 42.78  | 68.75  |
|RJB-AGCN| 89.85 | 95.20 | 85.47  | 86.05  | 42.22  | 63.92  |
|FR-AGCN | 90.46 | 95.83 | 86.60  | 86.99  | 43.98  | 69.50  |

# Data Preparation

 For 'NTU RGB+D':

 - Download the raw data from 'NTU-RGB+D' (http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp).
 - Then put them under the data directory:
 
        -data\  
          -nturgbd_raw\  
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt

 - Preprocess the data with
  
    `python data_gen/ntu_gendata.py`

 - Generate the forward data with: 
    
    `python data_gen/gen_forward_data.py`

 - Generate the reverse data with: 

    `python data_gen/gen_reverse_data.py`
	
 - Generate the bone data with: 
 
    `python data_gen/gen_forward_bone_data.py`
    
    `python data_gen/gen_reverse_bone_data.py`

 For 'NTU RGB+D 120':

 - Download the raw data from 'NTU-RGB+D 120' (http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp).
 - Then put them under the data directory:
 
        -data\  
          -nturgbd_raw\  
            -nturgb+d120_skeletons\
              ...
            -NTU_RGBD120_samples_with_missing_skeletons.txt

 - Preprocess the data with
  
    `python data_gen/ntu120_gendata_xsub_train.py`
    
    `python data_gen/ntu120_gendata_xsub_val.py`
    
    `python data_gen/ntu120_gendata_xset_train.py`
    
    `python data_gen/ntu120_gendata_xset_val.py`
	
 - Generate the forward data with: 
    
    `python data_gen/gen_forward_data_ntu120_xsub_train.py`
    
    `python data_gen/gen_forward_data_ntu120_xsub_val.py`
    
    `python data_gen/gen_forward_data_ntu120_xset_train.py`
    
    `python data_gen/gen_forward_data_ntu120_xset_val.py`

 - Generate the reverse data with: 

    `python data_gen/gen_reverse_data_ntu120_xsub_train.py`
    
    `python data_gen/gen_reverse_data_ntu120_xsub_val.py`
    
    `python data_gen/gen_reverse_data_ntu120_xset_train.py`
    
    `python data_gen/gen_reverse_data_ntu120_xset_val.py`

 - Generate the bone data with: 
 
    `python data_gen/gen_forward_bone_data_ntu120.py`
    
    `python data_gen/gen_reverse_bone_data_ntu120.py`

For 'UAV-Human':

 - Download the raw data from 'UAV-Human' (https://github.com/SUTDCV/UAV-Human).
 - For the CSv2 benchmark of UAV-HUman, you may have to classify the training set and the testing set according to the ID.
 - We experimented with two benchmarks in two folders，i.e., UAVAGCN and UAVAGCN1.
 - It is recommended to use the method developed by the author for preprocessing, but the preprocess file needs to be replaced (https://github.com/SUTDCV/UAV-Human/tree/master/uavhumanposetools).
 - Pay attention to the setting of the number of joint points and the maximum number of frames.
 - Then put them under the data directory:
 
        -data\  
          -uav\  
            -train_data.npy
            -train_label.pkl
            -val_data.npy
            -val_label.pkl

 - Preprocess the data with

    `python data_gen/gen_forward_data.py`

    `python data_gen/gen_reverse_data.py`

    `python data_gen/gen_forward_bone_data.py`

    `python data_gen/gen_reverse_bone_data.py`

# Training & Testing

 For 'NTU RGB+D':
 
 - X-sub (Cross-Subject):
 
    `python main.py --config ./config/nturgbd-cross-subject/train_forward.yaml`
    
    `python main.py --config ./config/nturgbd-cross-subject/train_reverse.yaml`
    
    `python main.py --config ./config/nturgbd-cross-subject/train_forward_bone.yaml`
    
    `python main.py --config ./config/nturgbd-cross-subject/train_reverse_bone.yaml`
    
    `python main.py --config ./config/nturgbd-cross-subject/test_forward.yaml`
    
    `python main.py --config ./config/nturgbd-cross-subject/test_reverse.yaml`
    
    `python main.py --config ./config/nturgbd-cross-subject/test_forward_bone.yaml`
    
    `python main.py --config ./config/nturgbd-cross-subject/test_reverse_bone.yaml`
 
 - X-view (Cross-View):
 
    `python main.py --config ./config/nturgbd-cross-view/train_forward.yaml`
    
    `python main.py --config ./config/nturgbd-cross-view/train_reverse.yaml`
    
    `python main.py --config ./config/nturgbd-cross-view/train_forward_bone.yaml`
    
    `python main.py --config ./config/nturgbd-cross-view/train_reverse_bone.yaml`

    `python main.py --config ./config/nturgbd-cross-view/test_forward.yaml`
    
    `python main.py --config ./config/nturgbd-cross-view/test_reverse.yaml`
    
    `python main.py --config ./config/nturgbd-cross-view/test_forward_bone.yaml`
    
    `python main.py --config ./config/nturgbd-cross-view/test_reverse_bone.yaml`
 
 - Finally combine the generated scores with: 

    `python ensemble_4s.py --datasets ntu/xsub`
    
    `python ensemble_4s.py --datasets ntu/xview`

 For 'NTU RGB+D 120':

 - Take the benchmark X-sub (Cross-Subject) as an example:

    `python main.py --config ./config/nturgbd120-cross-subject/train_forward.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-subject/train_reverse.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-subject/train_forward_bone.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-subject/train_reverse_bone.yaml`

    `python main.py --config ./config/nturgbd120-cross-subject/test_forward.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-subject/test_reverse.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-subject/test_forward_bone.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-subject/test_reverse_bone.yaml`

 - Perform similar operations on another benchmark X-set (Cross-Setup).

    `python main.py --config ./config/nturgbd120-cross-setup/train_forward.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-setup/train_reverse.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-setup/train_forward_bone.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-setup/train_reverse_bone.yaml`

    `python main.py --config ./config/nturgbd120-cross-setup/test_forward.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-setup/test_reverse.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-setup/test_forward_bone.yaml`
    
    `python main.py --config ./config/nturgbd120-cross-setup/test_reverse_bone.yaml`

 - Finally combine the generated scores with: 

    `python ensemble120_4s.py --datasets ntu/xsub`
    
    `python ensemble120_4s.py --datasets ntu/xset`
    
 For 'UAV-Human':
 
 - Take the benchmark CSv1 as an example (Note that we put the two benchmarks in two **different** folders.):
 
    `python main.py --config ./config/uav/train_forward.yaml`
    
    `python main.py --config ./config/uav/train_reverse.yaml`
    
    `python main.py --config ./config/uav/train_forward_bone.yaml`
    
    `python main.py --config ./config/uav/train_reverse_bone.yaml`

    `python main.py --config ./config/uav/test_forward.yaml`
    
    `python main.py --config ./config/uav/test_reverse.yaml`
    
    `python main.py --config ./config/uav/test_forward_bone.yaml`
    
    `python main.py --config ./config/uav/test_reverse_bone.yaml`
    
 - Finally combine the generated scores with:     
 
    `python ensemble_uav_4s.py  --datasets uav`

# Acknowledgements

 This work is based on

 2s-AGCN (https://github.com/lshiwjx/2s-AGCN)

 Thanks to the original authors for their work! Our work is only the improvement of the data preprocessing part based on it. However, we hope that the research content of this forward and inverse sequences can be inspiring for someone.

 Meanwhile, we are very grateful to the creators of these three datasets, i.e., NTU RGB+D 60, NTU RGB+D 120, UAV-Human. Your selfless work has made a great contribution to the computer vision community!

# Contact

 If you find that the above description is not clear, or you have other issues that need to be communicated when conducting the experiment,
 please leave a message on github. Besides, we look forward to discussions about skeleton-based action recognition.
 Feel free to contact me via email:
     `20194229016@stu.suda.edu.cn`
 
 I expect to graduate in July 2022, feel free to contact me via email after that:
     `741465026@qq.com`
