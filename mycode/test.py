import numpy as np
import skimage as io
import torch
import torch.nn.functional as F
from train import Variable


def get_patches(img_path):
    ### set params and read image
    patch_size = 128
    step = 64
    n_channels = 1
    n_classes = 4
    img = io.imread(img_path, as_gray=True)
    if img.dtype == np.uint16:
        img = img.astype(np.int32)
    
    # since I'm testing with 128x128 images, upscale to 1024x1024
    img = torch.from_numpy(img.astype(np.float32))
    img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
    img = img.squeeze(0).squeeze(0)
    
    ### initialize weight map and list of patches
    weight_map = np.zeros((img.shape[0], img.shape[1], n_classes))
    patches = []
    
    ### slide moving window through the image and save the patches
    ix, iy = patch_size, patch_size
    while iy <= img.shape[0]: # < or <=??
        while ix <= img.shape[1]:
            patch = img[iy-patch_size:iy, ix-patch_size:ix]
            patches.append(patch)
            weight_map[iy-patch_size:iy, ix-patch_size:ix] += 1.0
            ix += step
        ix = patch_size
        iy += step
    
    patches = np.array(patches)
    patches = np.reshape(patches, 
                        (patches.shape[0], n_channels, patch_size, patch_size))
    
    return weight_map, torch.FloatTensor(patches)


def predict_on_patches(model, patches):
    max_batch_size = 16
    remaining_patches = len(patches)
    a_predictions = np.zeros((len(patches), 4))
    
    p = 0
    while remaining_patches > 0:
        print(remaining_patches)
        batch_size = min(max_batch_size, remaining_patches)
        
        output = model(Variable(patches.narrow(0, p, batch_size), requires_grad=False)) #.cuda()
        output = (output * 100.).round() / 100. # round off results
        output[:,0] = output[:,0].round() # round a_0 to 0 or 1
        a_predictions[p:p+batch_size] = output.data.cpu().numpy()
        
        p += batch_size
        remaining_patches -= batch_size
        
    return a_predictions


def compose_psf_map(a_predictions, weight_map, patch_size, step):
    img_height, img_width = weight_map.shape[0], weight_map.shape[1]
    
    psf_map = np.zeros((weight_map.shape[0], weight_map.shape[1], a_predictions.shape[1]))

    ix, iy = patch_size, patch_size
    p = 0
    while iy <= img_height:
        while ix <= img_width:
            psf_map[iy-patch_size:iy, ix-patch_size:ix] += a_predictions[p, :]
            ix += step
            p += 1
        ix = patch_size
        iy += step
    psf_map = psf_map / weight_map
    # psf_map[:,:,0] = np.round(psf_map[:,:,0])
    
    return psf_map
