# Camelyon17 - Multilevel feature fusion in digital pathology

Deep learning algorithms have proven to be efficient and accurate when detecting
metastases in hematoxylin and eosin-stained tissue, and their performance is comparable to
the level of an expert pathologist. Many of the tumor-detecting deep learning algorithms
focus on the local features that are in the small batches of images, which leaves out
potentially relevant features from the surroundings.[1] For example, the eosinophils are
characteristic in some parts of a tissue sample, whereas some parts can indicate a tumor or other
disease. Small image batches may not contain enough spatial information for this.

## Research questions
1. Does including information from the surrounding area, improve the performance of
deep learning tumor detection algorithm?
2. What features will a deep neural network focus on with different scales, when it is
trained to detect a tumor?

## Hypothesis
A deep neural network will learn to use information from a wider receptive field and this
improves the detection performance. High zoom level parts of the network will focus on the
detailed structures while the low zoom levels will focus more on regional structures.

## Methodology
1. Build one or more convolutional neural network (CNN) architectures that use
information from different zoom levels. Design one, otherwise similar architecture for
comparison that uses only data from the highest zoom level.
2. Split Cameleon17 training data set to training, validation, and test parts.
3. Sample Camelyon17 training and validation set. Sample evenly from tumor and
normal areas. Sample only from the tissue areas (Otsu or another thresholding
method).
4. Sample Cameleon17 test set covering all tissue parts. Overlapping sampling.
5. Train models and measure the performance on the validation set. Binary
classification metric: ROC AUC. Optimize architectures and training
hyper-parameters.
6. Predict probability heatmaps for the test set and threshold for tumor detection.
Measure mean IOU (area of overlap/area of a union) and compare to literature.
7. Analyze the models by visualizing the gradient class activation mappings (Grad-CAM) 
from the different zoom levels [2].
8. Transfer models to different cancer if annotated data is available, and measure the
performance. (not included here)

----------------------------------------

# Project

This section describes the project structure, software dependencies, and notebook contents.

### Environment used
- 64bit Ubuntu 16.04.6 LTS (Xenial Xerus) GNU/Linux (virtual)
- 2x Intel Xeon Platinum 8160 CPU @ 2.10GHz
- 4x Nvidia Tesla V100, 32GB
- 1510GB RAM
- 6.4T NVMe SSD

### Software used
- [PyTorch](https://pytorch.org/docs/stable/index.html) 1.1.0
- TorchVision 0.3.0
- [Fastai](https://docs.fast.ai/) 1.0.52
- [OpenSlide](https://openslide.org/download/) 3.4.1 (ASAP 1.8 depends on libopenslide)
- [ASAP](https://github.com/computationalpathologygroup/ASAP) 1.8 (1.9 does not support the Philips scanner TIFF file format of the `Center_4`)
- [OpenCV](https://opencv.org/) 4.1.0

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

## Hyperparam opt. and testing results
The scores are averages from 4 CV folds. Model folds are trained on 3 and tested on one of the train centers (0-3).

**Baseline models**

|id   |Model       | Description | AUC   |
|:--:|:-----------:|:-----------:|:----:|
|01  | DenseNet121 |10+10 epochs |93.02 |
|02  | DenseNet121 |8+4 epochs   |94.56 |
|03  | DenseNet121 |4+2 epochs   |94.82 |
|04  | DenseNet121 |4+2 epochs, Normalized   |94.89 |
|05  | DenseNet169 |4+2 epochs   |92.90 |
|06  | SENet154    |4+2 epochs   |95.73 |
|07  | InceptionResNetv2    |4+2 epochs   |96.30 |
|07N | InceptionResNetv2    |4+2 epochs, Normalized  |96.13 |
|08  | Se-ResNeXt101 32x4d    |4+2 epochs   |96.20 |
|08N | Se-ResNeXt101 32x4d    |4+2 epochs, Normalized   |96.30 |
|10  | Se-ResNeXt101 32x4d    |1 epoch   |97.27 |
|10N  | Se-ResNeXt101 32x4d    |1 epoch, Normalized   | 96.52 |

**Multilevel models**

|id   |Model (context)    |Model (focus) | Description | AUC   |
|:--:|:-----------:|:----------:|:-----------:|:----:|
|09  | ResNet18 |ResNet50 |1 epoch, lvls 3 & 0 |96.56 |
|11  | ResNet34 |ResNet10 |1 epoch ,lvls 3 & 0  |94.34 |
|12  | ResNet18 |ResNet50 |1 epoch, Normalized ,lvls 3 & 0  |96.96 |
|13  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 3 & 0  |97.26 |
|14  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 2 & 0  |97.43 |
|15  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 0 & 0  |96.15 |
|16  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 0 & 2 , context model pretrained with autoencoder |98.24 |
|17  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 0 & 2 , context model pretrained with autoencoder |98.13 |

## Test results
The scores are average from 3 replicates. Models are trained on all train centers (0-3) and tested on test center (4).

**Baseline models**

|id   |Model       | Description | AUC_avg | AUC_1 | AUC_2 | AUC_3 |
|:--:|:-----------:|:-----------:|:----:|:----:|:----:|:----:|
|10  | Se-ResNeXt101 32x4d    |1 epoch   |95.47 | 96.11 | 95.48 | 94.84 |
|10N  | Se-ResNeXt101 32x4d    |1 epoch, Normalized  | 95.84 | 95.98 | 95.87 | 95.67 |

**Multilevel models**

|id   |Model (context)    |Model (focus) | Description | AUC_avg | AUC_1 | AUC_2 | AUC_3 |
|:--:|:-----------:|:----------:|:-----------:|:----:|:----:|:----:|:----:|
|13  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 3 & 0  | 95.53 | 95.71 | 96.17 | 94.69 |
|14  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 2 & 0  | 95.99 | 95.25 | 95.83 | 96.89 |
|15  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 0 & 0  | 95.75 | 96.13 | 95.33 | 95.75 |
|16  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 0 & 2 , context model pretrained with autoencoder | - | - | - | - |
|17  | Se-ResNeXt50 |Se-ResNeXt101 |1 epoch, Normalized ,lvls 0 & 2 , context model pretrained with autoencoder | - | - | - | - |

-----------------------------------

## References

[1] B. E. Bejnordi, M. Veta, P. J. van Diest, B. van Ginneken, N. Karssemeijer, G. Litjens,
J. A. W. M. van der Laak, and the CAMELYON16 Consortium. (2017) Diagnostic
assessment of deep learning algorithms for detection of lymph node metastases in women
with breast cancer. JAMA . 318 (22):2199–2210. doi: 10.1001/jama.2017.14585

[2] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. (2017)
Grad-CAM: Visual explanations from deep networks via gradient-based localization.
arXiv:1610.02391
