# add ASAP path to sys to locate the multiresolutionimageinterface
import sys
sys.path.append('/opt/ASAP/bin')
# required libraries
import multiresolutionimageinterface as mir
import cv2
import numpy as np
import os
from tqdm import tqdm_notebook
import pandas as pd
from utils.subutils import patchAttributes
from utils.subutils import conversions

def getTissueMask(tifPath):
    maskPath = tifPath.replace('.tif', '_tissue_mask_ds16.npy')
    if (not os.path.isfile(maskPath)): return None
    return np.load(maskPath)

def getImage(tifPath):
    reader = mir.MultiResolutionImageReader()
    if (not os.path.isfile(tifPath)): return None
    return reader.open(tifPath)

def getAnnoMask(tifPath):
    reader = mir.MultiResolutionImageReader()
    maskPath = tifPath.replace('.tif', '_mask.tif')
    if (not os.path.isfile(maskPath)): return None
    return reader.open(maskPath)

def getPatchAndMasks(mr_image, mr_mask, tissue_mask,center, side=256):
    patch_bounds = conversions.center2Bounds(center)
    mask_bounds = conversions.center2Bounds(center, ds=16)
    
    channels = 3
    annoMask = np.zeros((side, side, channels), dtype=np.uint8)

    img = mr_image.getUCharPatch(int(patch_bounds[0]),
                                 int(patch_bounds[2]),
                                 side,
                                 side,
                                 0)
    
    tissueMask = tissue_mask[mask_bounds[2]:mask_bounds[3],mask_bounds[0]:mask_bounds[1]]
    if mr_mask is not None:
        annoMask = mr_mask.getUCharPatch(int(patch_bounds[0]),
                                     int(patch_bounds[2]),
                                     side,
                                     side,
                                     0)
    return img, tissueMask, np.array(annoMask)

def CreateDF(tifPath, overrideExisting=False, dirData = 'data/training/'):
    # get only the name without dir or file suffix
    fileNamePart = tifPath.replace('.tif','').replace(dirData, "")
    df_path = 'data/training/dataframes/' + fileNamePart.split('/')[1] + '.csv'
    
    if (os.path.isfile(df_path) and overrideExisting == False):
        print('Info - Dataframe file of {0} already exists - skipping'.format(tifPath))
        return
    
    tissue_mask = getTissueMask(tifPath)
    patch_centers = conversions.sample_centers(tissue_mask)

    print("Sliced WSI {1} to {0} pathes.".format(len(patch_centers), tifPath))
    
    # modify the global images
    mr_image = getImage(tifPath)
    mr_mask = getAnnoMask(tifPath)
    
    df = pd.DataFrame(columns=['patchId',
                               'fileName',
                               'center',
                               'patient',
                               'node',
                               'centerX',
                               'centerY',
                               'isTumor',
                               'tumorPercentage',
                               'tissuePercentage',
                               'meanHue',
                               'meanSaturation',
                               'meanValue'])
    
    split = tifPath.split('/')
    cnt = int(split[2].strip('center_'))
    splitpatient = split[3].split('_')
    patient = int(splitpatient[1])
    node = int(splitpatient[3].strip('.tif'))
    
    for c in tqdm_notebook(patch_centers, 'Patches...'):
        img,tissue,anno = getPatchAndMasks(mr_image, mr_mask, tissue_mask, c)
        isTumor_attr = patchAttributes.isTumor(anno)
        tumorPrc_attr = patchAttributes.tumorPercentage(anno)
        tissuePrc_attr = patchAttributes.tissuePercentage(tissue)
        colorMean_attr = patchAttributes.colorMean(img)
        (mean_h, mean_s, mean_v) = conversions.rgb2hsv(colorMean_attr)
        
        df = df.append({'patchId': str(patient)+str(0)+str(c[0]).zfill(7)+str(c[1]).zfill(7),
                       'fileName': tifPath,
                       'center': cnt,
                      'patient': patient,
                      'node': node,
                      'centerX':c[0],
                      'centerY':c[1],
                      'isTumor':isTumor_attr,
                      'tumorPercentage': int(tumorPrc_attr * 1000)/10,
                      'tissuePercentage': int(tissuePrc_attr * 1000)/10,
                      'meanHue': int(mean_h * 100)/100,
                      'meanSaturation': int(mean_s * 100)/100,
                      'meanValue': int(mean_v * 100)/100}, ignore_index=True)
        
    df.to_csv(df_path)

