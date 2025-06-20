# Melanoma Breslow Thickness Prediction with Test-Time Augmentations and Calibration-Aware Training

## Abstract
Breslow thickness is critical for melanoma prognosis, as it is
used to determine surgical margins and the need for sentinel lymph node
biopsy. While deep learning has shown promise in malignancy detection,
the automation of Breslow thickness prediction remains underexplored
due to the limited availability of well-annotated datasets.
In this study, we propose a convolutional neural network model to classify
melanoma Breslow thickness based on dermoscopic images into three (in
situ, thin, and thick) categories. A threshold between thin and thick of
0.8 mm is based on the sentinel lymph node biopsy recommendation. To
improve model robustness and calibration, we apply test-time augmenta-
tion and use calibration-aware training with focal loss to mitigate over-
and underconfident predictions. The models are trained and evaluated
using stratified 5-fold cross-validation on three datasets from different
sources.
Experimental results demonstrate that applying test-time augmenta-
tions increases balanced accuracy by 1.4%. Furthermore, combining test-
time augmentations with confidence-based sample rejection improves
balanced accuracy by 3.1%. The five-fold stratified cross-validation mod-
els achieve an average AUC of 0.811 ± 0.007 for multiclass classification.
Our findings highlight the potential of deep learning in assisting derma-
tologists with non-invasive Breslow thickness estimation, enabling better
clinical decision-making.

