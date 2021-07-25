# Color Spray

![](https://github.com/fengwang/color_spray/raw/master/assets/demo.png)


STEM images colorization using Deep Convolutional Neural Networks.

## Installing

__A__. Install and update using [pip](https://pip.pypa.io/en/stable/quickstart/):

```bash
pip3 install color-spray
```
Or
```bash
git checkout https://github.com/fengwang/color_spray.git
cd color_spray
python3 -m pip install -e .
```

__B__. Dowload pretrained model files from `https://drive.google.com/drive/folders/1dl-iNgROmSv71EpzNalh97zksKIBc90u?usp=sharing`, place them in your home folder, under path `.color_spray/model`.

## Usage


Command line:

```bash
color_spray INPUT_GRAY_IMAGE_PATH OUTPUT_RGB_IMAGE_PATH
```

Using Python API:

```python
# uncomment the follow three lines if you have a Nvidia GPU but you do not want to enable it.
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=''

from color_spray import color_spray
rgb_image = color_spray( './a_gray_image.png', './an_rgb_image.png' )

```

## Details

+ The training images are downloaded from [PEXEL](https://www.pexels.com/).

## License

+ BSD

