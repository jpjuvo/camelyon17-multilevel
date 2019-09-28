# Camelyon17 - Multilevel feature fusion in digital pathology

![Tumor instance segmentation](/img/instance_segmentation_demo.png)

*Metastasis instance segmentation that was produced with a multilevel model.*

## Introduction

Deep learning algorithms have proven to be efficient and accurate when detecting
metastases in hematoxylin and eosin-stained tissue, and their performance is comparable to
the level of an expert pathologist. Many of the tumor-detecting deep learning algorithms
focus on the local features that are in the small batches of images, which leaves out
potentially relevant features from the surroundings.[1] For example, the eosinophils are
characteristic in some parts of a tissue sample, whereas some parts can indicate a tumor or other
disease. Small image batches may not contain enough spatial information for considering the surroundings.

## Research questions
1. Does including information from the surrounding area, improve the performance of
deep learning tumor detection algorithm?
2. What features will a deep neural network focus on with different scales, when it is
trained to detect a tumor?

### Hypothesis
A deep neural network will learn to use information from a wider receptive field and this
improves the detection performance. High zoom level parts of the network will focus on the
detailed structures while the low zoom levels will focus more on regional structures.

## Methodology

### Environment
- 64bit Ubuntu 16.04.6 LTS (Xenial Xerus) GNU/Linux (virtual)
- 2x Intel Xeon Platinum 8160 CPU @ 2.10GHz
- 4x Nvidia Tesla V100, 32GB
- 1510GB RAM
- 6.4T NVMe SSD

### Software
- [PyTorch](https://pytorch.org/docs/stable/index.html) 1.1.0
- TorchVision 0.3.0
- [Fastai](https://docs.fast.ai/) 1.0.52
- [OpenSlide](https://openslide.org/download/) 3.4.1 (ASAP 1.8 depends on libopenslide)
- [ASAP](https://github.com/computationalpathologygroup/ASAP) 1.8 (1.9 does not support the Philips scanner TIFF file format of the `Center_4`)
- [OpenCV](https://opencv.org/) 4.1.0

### Preprocessing
[Chameleon17](https://camelyon17.grand-challenge.org/Data/) training data set was divided by medical centers to test (center_4) and train parts (center_0, center_1, center_2, center_3). Whole slide image (WSI) tissue areas were sampled to 256x256 overlapping tiles, where the corners of tile were centers of neighboring tiles. Otsu thresholding was used for finding the tissue areas.

Tumor coverage percentages were calculated for each tile, and a 75% threshold was selected for labeling a tile as a tumor or normal. Tiles were undersampled from each medical center so that tumor and normal tiles were represented in equal amounts.

Image crops of size 256x256 were sampled from each tile's center point in 1, 2, 4 and 8 -pixel downsampling rates. Each downsampling crop had the same center point as the tile.

![downsampling](/img/tumor_label.png)

*Downsampling rates with green tumor area annotation. Threshold of 75% in downsampling=1 is used for deciding the tumor label.*

Normalized copies were made from each image crop and normalization was done using color deconvolution to separate haematoxylin and eosin components, and normalizing their amounts using a reference. 

<img src="/img/normalization.png" alt="normalization" width="400"/>

*Tissue samples before and after the staining normalization in two leftmost columns. Two rightmost columns show the separated haematoxylin and eosin stains.*

### Baseline models
Baseline models were trained to do binary classification (tumor/normal) from crop images and ROC AUC was used as the performance metric. Medical center-fold cross-validation was used for searching the optimal  CNN architecture, learning rate cycle, and training augmentations. Cross-validation was done with the training set of 4 medical centers and only crop images of pixel samplingrate1 were used. The effectiveness of stain normalization was determined by training models with either normalized or original images.

### Multilevel models
Multi-input (multilevel) models were assembled from the good performing baseline model architectures. These models took two different downsampling rate images as input. Both downsampling rates had the same center, so the models were looking at the same spot from two different zoom scales. Multilevel models consisted of two separate CNN base architectures. One for the context (lower zoom and wider receptive field) and a deeper architecture for the focus (highest zoom). The output vectors from the last convolutional layers of both architectures were combined with linear layers to produce a single binary output.

<img src="/img/multilevel_architecture_two_levels.png" alt="Multilevel architecture" width="500"/>

*Multilevel architecture*

### Testing
Five replicates of each of the best performing baseline and multilevel models were trained on all training folds. Their performance was measured in the test fold.

Three of the tumor region containing test set WSI's were re-sampled covering the whole tissue region. Best performing models were used for generating tile-level tumor-probability heatmaps. Tumor regions were thresholded from the heatmaps, and each model's threshold value was selected from the highest F-0.5 score in the training folds.

## Results

### Training folds optimization

Leave-one-center-out cross-validation.

-**Fold_0**: Train=`{center_1,center_2,center_3}`, Validation=`{center_0}`
-**Fold_1**: Train=`{center_0,center_2,center_3}`, Validation=`{center_1}`
-**Fold_2**: Train=`{center_0,center_1,center_3}`, Validation=`{center_2}`
-**Fold_3**: Train=`{center_0,center_1,center_2}`, Validation=`{center_3}`

*id suffix N* = Trained and tested on normalized data

*id suffix A* = Trained on heavily color augmented data

![Training all](/img/train_results_all.png)

![Training avg](/img/train_results_avg.png)

*Average AUCs from all of the four folds. The red dotted line is the best baseline average AUC*

### Test fold

- **Train**=`{center_0,center_1,center_2,center_3}`, Test=`{center_4}`

*id suffix N* = Trained and tested on normalized data

*id suffix A* = Trained on heavily color augmented data

![Test all](/img/test_results_all.png)

*The red dotted line is the best baseline run. The two baseline models use SE-ResNeXt101 32x4d architecture. All multilevel models use SE-ResNeXt50 32x4d as the context model and SE-ResNeXt101 32x4d as the focus model.*

![Test avg](/img/test_results_avg.png)

*The red dotted line is the best average baseline AUC*

### Tumor WSI's from the test fold 

Tumor masks were produced from three test fold WSI's that had tumor regions: `patient_081_node_4`, `patient_088_node_1` and `patient_099_node_4`.

![Test WSI AUC](/img/test_wsi_auc.png)
![Test WSI IOU](/img/test_wsi_iou.png)
![Test WSI DICE](/img/test_wsi_dice.png)

<img src="/img/multilevel-19-heatmap.png" alt="Multilevel tumor mask" width="600"/>

*Tumor segmentation of the three patient WSI's (`patient_081_node_4`, `patient_088_node_1` and `patient_099_node_4`) with the model 19A*

----------------------------------------

## Project

This section describes the project structure and notebook contents.

## Project structure
This project assumes that the [Camelyon17 training data set](https://camelyon17.grand-challenge.org/Data/) is downloaded and unzipped in the following way:

```
data/
    |_annotations/
        |_patient_004_node.xml
        |_...
    |_training/
        |_center_0
            |_patient_000.tif
            |_...
        |_...
    |_stage_labels.csv
```

## Notebooks
These should be run in order as the preprocessing steps are prerequisites for the later notebooks.

1. **Preprocessing** - Convert lesion annotations to masks
  - Camelyon17 annotations are stored in polygon representations (xml). This notebook converts them to tif pixel image masks where value of 1 means tumor and 2 means normal.
2. **Preprocessing** - View tumor annotations and create tissue masks
  - 16 times downsampled tissuemasks are stored as binary (0-background, 255-tissue) uint8 numpy arrays from each WSI.
3. **Preprocessing** - Create dataframes
  - Dataframe contains center coordinates, tissue percentage, tumor percentage, file information, and label of all the tissue samples.
4. **Statistics** - Patch stats
5. **Dataset** - Sampling splits
6. **Dataset** - Creating patches
7. **Normalization** - Normalize H&E staining in patches to compare models with, and without normalization.
8. **Baseline** - Baseline models - hyperparameter optimization
9. **Multilevel** - Multilevel models - hyperparameter optimization
10. **Pretraining modules** - Pretraining multilevel CNN modules
11. **Pretrained multilevel** - Multilevel models with autoencoder pretrained context encoders
12. **Test** - Test set performance
13. **Threshold selection** - Search for the best WSI heatmap binary threshold with the training set.
14. **WSI heatmap** - Tumor heatmap for the test set tumor WSI's
15. **Conclusion** - Overview and analysis of the results

-----------------------------------

## References

[1] B. E. Bejnordi, M. Veta, P. J. van Diest, B. van Ginneken, N. Karssemeijer, G. Litjens,
J. A. W. M. van der Laak, and the CAMELYON16 Consortium. (2017) Diagnostic
assessment of deep learning algorithms for detection of lymph node metastases in women
with breast cancer. JAMA . 318 (22):2199–2210. doi: 10.1001/jama.2017.14585

[2] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. (2017)
Grad-CAM: Visual explanations from deep networks via gradient-based localization.
arXiv:1610.02391
