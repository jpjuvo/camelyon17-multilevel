import os
import cv2
import numpy as np

def normalizeStaining(img_path, saveDir='data/norm_patches/', Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images. 
    
    Input:
        img_path: Input image path
        saveDir (String): Normalized image save directory (default = 'data/norm_patches/')
        Io: (optional) transmitted light intensity
        
    Output:
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
    
    #extract name for the savefile
    base=os.path.basename(img_path)
    name_wo_ext = os.path.splitext(base)[0]
    fn = os.path.join(saveDir, name_wo_ext)
    
    # skip if this file already exists
    #if (os.path.isfile(fn+'.png')):
    #    return
             
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    rimg = np.reshape(img.astype(np.float), (-1,3))
    
    # calculate optical density
    OD = -np.log((rimg+1)/Io)
    
    # remove transparent pixels
    ODhat = np.array([i for i in OD if not any(i<beta)])
        
    # compute eigenvectors
    eigvals, eigvecs = None, None
    try:
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    except AssertionError:
        print('Failed to normalize {0}, copying this to output file unaltered.'.format(img_path))
        cv2.imwrite(fn+'.png', img)
        return
    except np.linalg.LinAlgError:
        print('Eigenvalues did not converge in {0}, copying this to output file unaltered.'.format(img_path))
        cv2.imwrite(fn+'.png', img)
        return
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    C2 = np.array([C[:,i]/maxC*maxCRef for i in range(C.shape[1])]).T
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
        
    Inorm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(fn+'.png', Inorm)
    return