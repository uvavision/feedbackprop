## [Feedback-prop: Convolutional Neural Network Inference under Partial Evidence](https://arxiv.org/pdf/1710.08049.pdf)
[Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), [Kota Yamaguchi](http://vision.is.tohoku.ac.jp/~kyamagu/), [Vicente Ordonez](http://vicenteordonez.com/), CVPR 2018

For more details, please refer to [this paper](https://arxiv.org/pdf/1710.08049.pdf) or email tianlu@virginia.edu
### Abstract
We propose an inference procedure for deep convolutional neural networks (CNNs) when partial evidence is available. Our method consists of a general feedback-based propagation approach (feedback-prop) that boosts the prediction accuracy for an arbitrary set of unknown target labels when the values for a non-overlapping arbitrary set of target labels are known. We show that existing models trained in a multi-label or multi-task setting can readily take advantage of feedback-prop without any retraining or fine-tuning. Our feedback-prop inference procedure is general, simple, reliable, and works on different challenging visual recognition tasks. We present two variants of feedbackprop based on layer-wise and residual iterative updates. We experiment using several multi-task models and show that feedback-prop is effective in all of them. Our results unveil a previously unreported but interesting dynamic property of deep CNNs. We also present an associated technical approach that takes advantage of this property for inference under partial evidence in general visual recognition tasks.

### Requirements
- Python 2.7
- PyTorch 0.3.1
- Numpy, scikit-learn, Pandas(optional)

### Demo
-  Multi-label Image Annotation(COCO)
Please download [COCO](http://cocodataset.org/#home) dataset first. Specifically, 2014 train, val and test splits are used in the paper. To prepare a well trained multi-label model for feedback-prop, you can either train a new model by running [train.py](https://github.com/uvavision/feedbackprop/blob/master/coco_multilabel/train.py) in [coco_multilabel](https://github.com/uvavision/feedbackprop/blob/master/coco_multilabel) folder or directly download the pretrained [model](http://www.cs.virginia.edu/~tw8cb/files/model_best.pth.tar). We recommend to train a multi-label model by first fixing all CNN layers for a few epochs and then finetuning the model end-to-end. To apply feedback-prop, please go through [COCO-Feedback-prop.ipynb](https://github.com/uvavision/feedbackprop/blob/master/coco_multilabel/COCO-Feedback-prop.ipynb).

### Citing
If you find our paper/code useful, please consider citing:

```
@InProceedings{feedbackprop_CVPR_2018,
author = {Wang, Tianlu and Yamaguchi, Kota and Ordonez, Vicente},
title = {Feedback-prop: Convolutional Neural Network Inference under Partial Evidence},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
