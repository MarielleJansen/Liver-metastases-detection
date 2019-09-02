# Liver-metastases-detection

Code to train and test a dual pathway P-net CNN. The CNN takes dynamic contrast enhanced MR and diffusion weighted MR images as input. Each pathway of the CNN processes one MR sequence and at the end of the CNN the feature maps are concatenated.
A liver mask can be used to exclude the region outside the liver to reduce false positives.

Python 3.5 code using keras and tensorflow.
The training and validation data have been pre-processed, as described in the paper.

This code is used for: M.J.A. Jansen, H.J. Kuijf, M. Niekel, W.B. Veldhuis, F.J. Wessels, M.A. Viergever, and J.P.W. Pluim, “Liver segmentation and metastases detection in MR images using convolutional neural networks”, J Medl Imaging, 2019
