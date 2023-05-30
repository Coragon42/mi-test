import math
import skimage as sk
from scipy import ndimage as ndi
from scipy import optimize as opt
from scipy.stats import entropy
import numpy as np
from matplotlib import pyplot as plt

def normalized_mutual_information(A, B):
    '''Returns NMI between two images.'''
    hist, bin_edges = np.histogramdd([np.ravel(A), np.ravel(B)], bins=100)
    hist /= np.sum(hist) #normalization
    H_A = entropy(np.sum(hist, axis=0))
    H_B = entropy(np.sum(hist, axis=1))
    H_AB = entropy(np.ravel(hist))
    return (H_A+H_B)/H_AB

def cost_nmi(params, reference, target):
    '''Uses NMI as a cost function to be minimized elsewhere.'''
    transformed = sk.transform.warp(target, make_rigid_transform(params), order=3)
    return -normalized_mutual_information(reference, transformed)

def gaussian_pyramid(img, max_layer=4, downscale = 2):
    '''Returns array of pyramid levels, from lowest to highest resolution.
    Default max_layer is max possible number of levels.'''
    pyramid = [img]
    layer = 0
    while layer != max_layer:
        layer += 1
        prev_shape = img.shape
        blurred = ndi.gaussian_filter(img, sigma=2*downscale/6.0)
        img = sk.transform.resize(blurred, \
                                  tuple(d/float(downscale) for d in img.shape), \
                                  order=1, mode='reflect', cval=0, anti_aliasing=False)
        if img.shape == prev_shape:
            break
        pyramid.append(img)
    return reversed(pyramid)

def make_rigid_transform(params):
    '''Reformats rigid transformation parameters for skimage use.'''
    rot, tcol, trow = params
    return sk.transform.SimilarityTransform(rotation=rot,translation=(tcol,trow))

def align(reference, target, cost=cost_nmi, max_layer = 4, downscale = 2, method='BH'):
    '''Finds transformation that maps reference to target image.'''
    pyramid_reference = gaussian_pyramid(reference, max_layer=max_layer)
    pyramid_target = gaussian_pyramid(target, max_layer=max_layer)
    params = np.zeros(3)
    for n,(ref,tgt) in zip(np.arange(max_layer,0,-1),zip(pyramid_reference,pyramid_target)):
        params[1:] *= downscale
        if method == 'BH':
            res = opt.basinhopping(cost, params, minimizer_kwargs={'args':(ref,tgt)})
            if n <= 4: # basin-hopping is too slow at full resolution
                method = 'Powell'
        else:
            res = opt.minimize(cost, params, args=(ref,tgt), method='Powell')
        params = res.x
    return make_rigid_transform(params)

def main():
    astronaut = sk.color.rgb2gray(sk.data.astronaut())
    altered = ndi.shift(sk.transform.rotate(sk.util.invert(astronaut),13),(-50,10))
    tf = align(astronaut, altered)
    corrected = sk.transform.warp(altered, tf, order=3)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(astronaut)
    ax0.set_title('Original')
    ax1.imshow(altered)
    ax1.set_title('Altered')
    ax2.imshow(corrected)
    ax2.set_title('Registered')
    for ax in (ax0, ax1, ax2):
        ax.axis('off')
    plt.show()

main()