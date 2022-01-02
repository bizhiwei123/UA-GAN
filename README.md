# UA-GAN
Uncertainty and Attention based GAN for multimodal image translation
## Environment
```
python              3.8.10
pytorch             1.8.1
torchvision         0.9.1
tqdm                4.62.1
numpy               1.20.3
SimpleITK           2.1.0
scikit-learn        0.24.2
opencv-python       4.5.3.56
easydict            1.9
tensorboard         2.5.0
Pillow              8.3.1
```
## Datasets
Download the datasets from the official way and rearrange the files to the following structure.
The dataset path can be modified in the UA-GAN/options/\*.yaml file.
### BraTS2020
```
MICCAI_BraTS2020_TrainingData
├── flair
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_002_flair.nii.gz
│   ├── BraTS20_Training_003_flair.nii.gz
│   ├── ...
├── t2
│   ├── BraTS20_Training_001_t2.nii.gz
│   ├── BraTS20_Training_002_t2.nii.gz
│   ├── BraTS20_Training_003_t2.nii.gz
│   ├── ...
├── t1
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_002_t1.nii.gz
│   ├── BraTS20_Training_003_t1.nii.gz
│   ├── ...
├── t1ce
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_002_t1ce.nii.gz
│   ├── BraTS20_Training_003_t1ce.nii.gz
│   ├── ...
```

### RaFD
For RaFD dataset, you just need to follow the original dataset with the image name like 'Rafd090_01_Caucasian_female_angry_frontal.jpg'. We only select the images shooting from frontal direction (Rafd090) for training and testing.

## Train
Edit the .yaml file of the corresponding dataset for training configuration and run the following command to train our model.
```
python train.py options/brats.yaml
```

## Test
Edit the .yaml file of the corresponding dataset for testing configuration and run the following command to test.
```
python test.py options/brats.yaml
```
