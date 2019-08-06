import numpy as np
import colorsys

def sample_centers(tissue_mask, mask_downscale=16, sample_side=256, focus_width_percentage=0.25, padding_percentage=0.01):
    mask_width, mask_height = tissue_mask.shape[:2]
    side = sample_side / mask_downscale
    padding_width = mask_width*padding_percentage
    padding_height = mask_height*padding_percentage
    half_focus = int(sample_side*focus_width_percentage / mask_downscale)
    sample_centers = []
    
    for i in range(int(mask_width // side)):
        for j in range(int(mask_height // side)):
            for sub_shift in [0, 0.5]:
                x = int((i+sub_shift) * side)
                y = int((j+sub_shift) * side)
                min_x = int(max(0, x - half_focus))
                max_x = int(min(x + half_focus, mask_width - 1))
                min_y = int(max(0, y - half_focus))
                max_y = int(min(y + half_focus, mask_height - 1))
                
                if(min_x < padding_width or max_x > mask_width-padding_width): continue
                if(min_y < padding_height or max_y > mask_height-padding_height): continue
                
                if(tissue_mask[min_x:max_x, min_y:max_y].sum() > 0):
                    sample_centers.append(np.array([x, y]))
                    
    # undo mask downscale to coordinates
    sample_centers = np.array(sample_centers) * mask_downscale
    return sample_centers

# Colorspace conversion
def rgb2hsv(rgb):
    return colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])

def center2Bounds(center,ds=1, side=256):
    ''' Get an array of [x1,x2,y1,y2] from a center point in a set downsampling scale
    Args:
        center (int array of x,y): center point coordinate
        ds (int): downsampling scale
        side (int): size of the box side (default 256)
    '''
    assert center.shape[0] == 2, "Invalid center point shape. Got {0} but expected (2,)".format(center.shape)
    half_side = int((side / ds) // 2)
    return np.array([center[1]//ds-half_side,
                    center[1]//ds+half_side,
                    center[0]//ds-half_side,
                    center[0]//ds+half_side], dtype=np.int32)