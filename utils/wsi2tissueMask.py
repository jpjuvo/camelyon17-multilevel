# add ASAP path to sys to locate the multiresolutionimageinterface
import sys
sys.path.append('/opt/ASAP/bin')
# required libraries
import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm_notebook

## This function is adapted from a digital pathology pipeline code of Mikko Tukiainen
def make_tissue_mask(slide, mask_level = 4, morpho=None, morpho_kernel_size=5, morpho_iter=1, median_filter=False, return_original = False): 
    ''' make tissue mask
        return tissue mask array which has tissue locations (pixel value 0 -> empty, 255 -> tissue)
    Args:
        slide (MultiResolutionImage): MultiResolutionImage slide to process
        mask_level (int): defines the level of zoom at which the mask be created (default 4)
        morpho (cv2.MORPHO): OpenCV morpho flag, Cv2.MORPHO_OPEN or Cv2.MORPHO_CLOSE (default None)
        morpho_kernel_size (int): kernel size for morphological transformation (default 5)
        morpho_iter (int): morphological transformation iterations (default=1)
        median_filtern (bool): Use median filtering to remove noise (default False)
        return_original (bool): return also the unmasked image
    '''
    
    ## Read the slide
    ds = slide.getLevelDownsample(mask_level)
    original_tissue = slide.getUCharPatch(0,
                                         0,
                                         int(slide.getDimensions()[0] / float(ds)),
                                         int(slide.getDimensions()[1] / float(ds)),
                                         mask_level)
    
    ## Determine the mask
    tissue_mask = cv2.cvtColor(np.array(original_tissue), cv2.COLOR_RGBA2RGB)
    tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2HSV)
    tissue_mask = tissue_mask[:, :, 1]
    _, tissue_mask = cv2.threshold(tissue_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    if morpho is not None:
        kernel = np.ones((morpho_kernel_size,morpho_kernel_size), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, morpho, kernel, iterations = morpho_iter)
        
    if median_filter:
        tissue_mask = cv2.medianBlur(tissue_mask, 15)
    
    tissue_mask = np.array(tissue_mask, dtype=np.uint8)

    if return_original:
        return tissue_mask, original_tissue
    else:
        return tissue_mask
    
def CreateTissueMask(tifPath, override_existing=True, dirData='data/training/'):
    
    # get only the name without dir or file suffix
    fileNamePart = tifPath.replace('.tif','').replace(dirData, "")
    
    # Skip if this mask is already found
    maskPath = tifPath.replace('.tif', '_tissue_mask_ds16.npy')
    if (os.path.isfile(maskPath) and override_existing==False):
        print('Info - Tissue mask file of {0} already exists - skipping'.format(tifPath))
        return
    
    # create tissue mask
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(tifPath)
    if(mr_image is None):
        print('Warning - Could not read {0} - skipping'.format(tifPath))
        return
    tissue_mask = make_tissue_mask(mr_image,
                                   mr_image.getBestLevelForDownSample(16), 
                                   morpho=cv2.MORPH_CLOSE,
                                   morpho_kernel_size=7,
                                   morpho_iter=2,
                                   median_filter=True)
    # tissue_mask is a binary array dtype.uint8  (16 times downsampled)
    np.save(maskPath, tissue_mask)
    print('Info - Created tissue mask {0}'.format(maskPath))