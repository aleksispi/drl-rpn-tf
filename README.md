# drl-RPN: Deep Reinforcement Learning of Region Proposal Networks for Object Detection
Official Tensorflow implementation of drl-RPN by Aleksis Pirinen (email: aleksis@maths.lth.se, webpage: [aleksispi.github.io](http://aleksispi.github.io)) and Cristian Sminchisescu ([webpage](http://www.maths.lth.se/matematiklth/personal/sminchis/)). The associated CVPR 2018 paper can be accessed [here](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pirinen_Deep_Reinforcement_Learning_CVPR_2018_paper.pdf). A video demonstrating this work can be seen [here](https://www.youtube.com/watch?v=XrszcAD-pnM).

The drl-RPN model is implemented on top of the publicly available TensorFlow VGG-16-based Faster R-CNN implementation by Xinlei Chen available [here](https://github.com/endernewton/tf-faster-rcnn). See also the associated technical report [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf), as well as the original Faster R-CNN paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497.pdf).

### Prerequisites
- A basic Tensorflow installation. The code follows r1.2 format.
- Python packages you might not have: cython, opencv-python, easydict (similar to py-faster-rcnn). For easydict make sure you have the right version (1.6 was used here).
- See also "Prerequisites" on [this page](https://github.com/endernewton/tf-faster-rcnn).

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/aleksispi/drl-rpn-tf.git
  ```
2. For steps 2-4, see "Installation" on [this page](https://github.com/endernewton/tf-faster-rcnn).

### Detection Performance
The current code supports VGG16 models. Exactly as for the Faster R-CNN implementation by Xinlei Chen, we report numbers using a single model on a single convolution layer, so no multi-scale, no multi-stage bounding box regression, no skip-connection, no extra input is used. The only data augmentation technique is left-right flipping during training following the original Faster R-CNN. 

We first re-ran some of the experiments reported [here](https://github.com/endernewton/tf-faster-rcnn) for Faster R-CNN, but training the models longer to obtain further performance gains for our baseline models. We got:
  - Train on VOC 2007+2012 trainval (*iterations*: 100k/180k) and test on VOC 2007 test (trained like [here](https://github.com/endernewton/tf-faster-rcnn), but for more iterations), **76.5**.
  - Train on VOC 2007+2012 trainval + 2007 test (*iterations*: 100k/180k) and test on VOC 2012 test, **74.0**.

The corresponding results when using our drl-RPN detector with exploration penalty 0.05 during inference (models trained over different exploration penalties, as described in Section 5.1.2 in the paper) and posterior class-probability adjustments:
  - Train on VOC 2007+2012 trainval (*iterations*: 90k/110k for core model, 80k/110k for posterior class-probability adjustment module) and test on VOC 2007 test (trained like [here](https://github.com/endernewton/tf-faster-rcnn), but for more iterations), **77.5**. Without posterior class-probability adjustments (np): 77.2. Average exploration (% RoIs forwarded per image on average): 28.0%. Average number of fixations per image: 5.6.
  - Train on VOC 2007+2012 trainval + 2007 test (*iterations*: 90k/110k, 80k/110k for posterior class-probability adjustment module) and test on VOC 2012 test, **74.9**. Without posterior class-probability adjustments (np): 74.6. Average exploration (% RoIs forwarded per image on average): 30.6%. Average number of fixations per image: 6.7.

**Tabular result representation**

| Model            | mAP - VOC 2007 | mAP - VOC 2012 |
| ---------------- | -------------- | -------------- |
| RPN              | 76.5           | 74.2           |
| drl-RPN          | 77.5           | 74.9           |
| drl-RPN (np)     | 77.2           | 74.6           |
| drl-RPN (12-fix) | 77.6           | 75.0           |

**Note**:
  - All settings are shared with that of Xinlei Chen for the things relating to Faster R-CNN.
  - See the code for any deviations from the [CVPR 2018 paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pirinen_Deep_Reinforcement_Learning_CVPR_2018_paper.pdf). Some important changes post-CVPR:
    - Training over different exploration-accuracy trade-offs is now the default model (as opposed to training for a fixed exploration penalty). Hence the default model allows for setting the exploration-accuracy trade-off during testing (c.f. Section 5.1.2 and Figure 6 in the paper). Turns out we only need two different exploration penalties (0.05 and 0.35 was used), but setting any other trade-off parameters during inference is possible.
    - Separation of rewards (Section 5.1.1 in the paper) does not yield accuracy gains for models trained over different exploration-accuracy trade-offs, so it is not used. See `reward_functions.py` for details.
    - The drl-RPN models are now much more fast to train than how it was done in the original paper (c.f. Section 5.2). Specifically, instead of sampling 50 search trajectories per image to estimate the policy gradient, we now run 50 search trajectories on 50 *different* images. This reduces training time by 5-10 times, yet we get results in the same ball park.

### Pretrained Models
All pretrained models (both Faster R-CNN baseline and our drl-RPN models) for the numbers reported above in *Detection Performance* is available on google drive: XYZ.
- drl-RPN trained on VOC 2007+2012 trainval: https://drive.google.com/open?id=1iK8fxp6no9g_-eZ2b2G0FRKV0cfUX53r
- drl-RPN trained on VOC 2007+2012 trainval + 2007 test: https://drive.google.com/open?id=1rNwmXLz9VCdK3s6dFqBH3rqpuVFtMLK7
- Faster R-CNN trained on VOC 2007+2012 trainval: https://drive.google.com/open?id=1UEvjBJwJFoGnv1DhrIsqmJWWWli8C9G4
- Faster R-CNN trained on VOC 2007+2012 trainval + 2007 test: https://drive.google.com/open?id=1ZFGuOitd8GA9QhqsdAYgc8Z0bILK0h3H

### Citation
If you find this implementation or our [CVPR 2018 paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pirinen_Deep_Reinforcement_Learning_CVPR_2018_paper.pdf) interesting or helpful, please consider citing:

    @article{pirinen2018deep,
        Author = {Aleksis Pirinen and Cristian Sminchisescu},
        Title = {Deep Reinforcement Learning of Region Proposal Networks for Object Detection},
        Journal = {IEEE Converence on Computer Vision and Pattern Recognition (CVPR)},
        Year = {2018}
    }
