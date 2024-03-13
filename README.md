# MATINEDepthSemanticSLAM

Project to combine SLAM with semantic segmentation, instance tracking, and scale estimation.

### Install Requirements (conda)
~~~
conda create --name matine python=3.10
conda activate matine
pip install -r dinov2/requirements.txt -r dinov2/requirements-extras.txt -r requirements.txt
~~~

*NOTE: Known problem with mmcv and mmsegmentation dependencies. Segmentation notebook does not work currently.

https://github.com/facebookresearch/dinov2/issues/196