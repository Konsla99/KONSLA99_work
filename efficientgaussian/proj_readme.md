https://xoft.tistory.com/48

Download colmap & setting
https://www.youtube.com/watch?v=UXtuigy_wYc&t=471s

colmap설치후 colmap 환경 변수등록

ImageMagik setting

https://imagemagick.org/script/download.php

ffmepg setting



1. 개인 영상 파일 준비

2. ffmepg를 통해 jpg파일 변환

Example
3. \data\dataset 폴더에  mydata.mp4준비
4. \data\dataset 폴더에 input dir 만들기

$ cd (root폴더)\data\dataset\input
$ ffmpeg -i (root폴더)\data\dataset\mydata.mp4 -qscale:v 1 -qmin 1 -vf fps=5 %04d.jpg

5. COLMAP
$ cd (root폴더)
$ python convert.py -s data/dataset

# train

python train_eval.py --config configs/efficient-3dgs.yaml -s <path to COLMAP or NeRF Synthetic dataset> -m <path to log directory> --save_ply --save_images


# rendering

python train_eval.py --config --config configs/efficient-3dgs.yaml -s <path to COLMAP or NeRF Synthetic dataset> -m <path to log directory of saved model> --skip_train
--save_images

or
python render.py -m <path to trained model> -s <path to COLMAP or NeRF Synthetic dataset>


# 360 rendering
python render_360.py -m <path to trained model> -s <path to COLMAP or NeRF Synthetic dataset>