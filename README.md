# UNet3plus_pth
UNet3+/UNet++/UNet, used in Deep Automatic Portrait Matting in Pytorth

## Dependencies

- Python 3.6
- PyTorch >= 1.1.0
- Torchvision >= 0.3.0
- future 0.18.2
- matplotlib 3.1.3
- numpy 1.16.0
- Pillow 6.2.0
- protobuf 3.11.3
- tensorboard 1.14.0
- tqdm==4.42.1

## Data
This model was trained from scratch with 18000 images (data augmentation by 2000images)
Training dataset was from [Deep Automatic Portrait Matting](http://www.cse.cuhk.edu.hk/leojia/projects/automatting/index.html).
Your can download in baidu cloud [http://pan.baidu.com/s/1dE14537](http://pan.baidu.com/s/1dE14537). Password: ndg8 
**For academic communication only, if there is a quote, please inform the original author!**

We augment the number of images by perturbing them withrotation and scaling. Four rotation angles{−45◦,−22◦,22◦,45◦}and four scales{0.6,0.8,1.2,1.5}are used. We also apply four different Gamma transforms toincrease color variation. The Gamma values are{0.5,0.8,1.2,1.5}. After thesetransforms, we have 18K training images. 

## Run locally
**Note : Use Python 3**

### Training

```shell script
> python train.py -h
usage: train.py [-h] [-g G] [-u U] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -g G, --gpu_id        Number of gpu
  -u U, --unet\_type    UNet type is unet/unet2/unet3
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 10.0)

```
By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively.

### Notes on memory
```bash
$ python train.py -g 0 -u v3 -e 200 -b 1 -l 0.1 -s 0.5 -v 15.0
```

### Prediction

You can easily test the output masks on your images via the CLI.

To predict a single image and save it:

```bash
$ python predict.py -i image.jpg -o output.jpg
```

To predict a multiple images and show them without saving them:

```bash
$ python predict.py -i image1.jpg image2.jpg --viz --no-save
```

```shell script

> python predict.py -h
usage: predict.py [-h] [--gpu_id 0]  [--unet\_type unet/unet2/unet3] [--model FILE] --input INPUT [INPUT ...] [--output INPUT [INPUT ...]] [--viz] [--no-save] [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  -g G, --gpu_id        Number of gpu
  --unet\_type, -u U    UNet type is unet/unet2/unet3
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```

## Reference

[[2015] U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI)](https://arxiv.org/pdf/1505.04597.pdf)

[[2018] UNet++: A Nested U-Net Architecture for Medical Image Segmentation (MICCAI)](https://arxiv.org/pdf/1807.10165.pdf)

[[2020] UNET 3+: A Full-Scale Connected UNet for Medical Image Segmentation (ICASSP 2020)](https://arxiv.org/pdf/2004.08790.pdf)