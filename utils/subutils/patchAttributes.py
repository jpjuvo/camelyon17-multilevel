import numpy as np
import cv2

def colorMean(tissue):
    (b,g,r,a) = cv2.mean(tissue)
    return np.array([r,g,b])/255

def isTumor(mask_level_0):
    '''
    Returns true if tumor mask has at least one pixel of tumor. 
    '''
    return (mask_level_0.max() > 0)

def tumorPercentage(mask_level_0):
    '''
    Returns the percentage of tumor (nonzero values) in tumor mask.
    '''
    area = mask_level_0.shape[0] * mask_level_0.shape[1]
    tumorPixels = np.count_nonzero(mask_level_0)
    channels = 3
    return tumorPixels / (area * channels)

def tissuePercentage(tissueMask):
    '''
    Returns the percentage of tissue (nonzero values) in a binary tissue mask.
    '''
    area = tissueMask.shape[0] * tissueMask.shape[1]
    tissuePixels = np.count_nonzero(tissueMask)
    return tissuePixels / area
