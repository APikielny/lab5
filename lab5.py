import numpy as np
import cv2
from scipy import ndimage
from skimage.transform import resize

def alpha_matte(im_size, feather_width, feather_location):

    output_im = np.ones((im_size[0], im_size[1]))
    feather_vals = np.arange(0, 1, 1/feather_width)
    feather_vals = np.vstack([feather_vals for i in range(im_size[0])])

    # All values on left before feather = 0 (black)
    output_im[:,0:(feather_location - feather_width//2)] = 0
    # All values on left before feather = 0 (white)
    output_im[:,(feather_location + feather_width//2):] = 1
    # Feather in the middle of the horizontal axis
    output_im[:,(feather_location - feather_width//2):
        (feather_location + feather_width//2)] = feather_vals
    
    # cv2.imshow('image', output_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return output_im

def blended_image(im1, im2, alpha_matte):
    im1 = np.array(im1)/255
    im2 = np.array(im2)/255
    alpha_matte = np.stack([alpha_matte,alpha_matte,alpha_matte], axis=-1)
    blended = (np.multiply(alpha_matte,im1) +
        np.multiply((1 - alpha_matte),im2))
    return blended

apple = cv2.imread('burt_apple.png')
orange = cv2.imread('burt_orange.png')
a = alpha_matte(apple.shape, 100, apple.shape[1]//2)
b = blended_image(apple, orange, a)

# cv2.imshow('Blended Output', b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def laplacian_pyramid(a, level, inList):
    if (a.shape[0] < 32) or (a.shape[1] < 32) or (level == 0):
        inList.append(a)
        return inList
        # return np.array(a)
    else:
        # Appy Gaussian filter and downsample images a and b, by factor of 2
        blurred_a = ndimage.gaussian_filter(a, sigma=5)
        details = a - blurred_a
        inList.append(details)
        return laplacian_pyramid(resize(blurred_a, (blurred_a.shape[0] // 2, blurred_a.shape[1] // 2), anti_aliasing=False), level-1, inList)
        # return np.array((details, laplacian_pyramid(resize(blurred_a, (blurred_a.shape[0] // 2, blurred_a.shape[1] // 2), anti_aliasing=False), level-1)))


def gaussian_pyramid(a, level, inList):
    if (a.shape[0] < 32) or (a.shape[1] < 32) or (level == 0):
        inList.append(a)
        return inList
    else:
        # Appy Gaussian filter and downsample images a and b, by factor of 2
        blurred_a = ndimage.gaussian_filter(a, sigma=5)
        inList.append(blurred_a)
        return gaussian_pyramid(resize(blurred_a, (blurred_a.shape[0] // 2, blurred_a.shape[1] // 2), anti_aliasing=False), level-1, inList)

def blend(img1, img2, mask):
    level = 3
    L1 = laplacian_pyramid(img1, level, [])
    L2 = laplacian_pyramid(img2, level, [])
    Gm = gaussian_pyramid(mask, level, [])
    cv2.imshow("L1.1", L1[0] /255.0)
    cv2.waitKey(0)
    cv2.imshow("Gm.1", Gm[0] /255.0)
    cv2.waitKey(0)


    L0 = []
    L0resized = []
    for l in range(level + 1):
        GmStacked = np.stack((Gm[l],Gm[l],Gm[l]), axis = -1)
        L0.append(GmStacked[l] * L1[l] + (1 - GmStacked[l] * L2[l]))
    for l in range(level + 1):
        L0resized.append(resize(L0[l], img1.shape, anti_aliasing = False))
    npL0 = np.array(L0resized)
    img = np.sum(npL0, axis = 0)

    return img
    

apple = cv2.imread('burt_apple.png')
orange = cv2.imread('burt_orange.png')
mask = alpha_matte(apple.shape, 100, apple.shape[1]//2)

blendedImg = blend(apple, orange, mask)
cv2.imshow("blended", blendedImg / 255.0)
cv2.waitKey(0)

def imalign_pyramid(a,b,level):
    '''
    Computes the offset of two images using an image pyramid and iterative
    refinement
    '''

    if (a.shape[0] < 32) or (a.shape[1] < 32):
        if a.shape[0] < a.shape[1]:
            shift_vector = np.asarray(imalign(a,b,a.shape[0]//2-1))
        else:
            shift_vector = np.asarray(imalign(a,b,a.shape[1]//2-1))
    elif level == 0:
        shift_vector = np.asarray(imalign(a,b,15))
    else:
        # Appy Gaussian filter and downsample images a and b, by factor of 2
        scaled_a = resize(a, (a.shape[0] // 2, a.shape[1] // 2), anti_aliasing=True, anti_aliasing_sigma=0.5)
        scaled_b = resize(b, (b.shape[0] // 2, b.shape[1] // 2), anti_aliasing=True, anti_aliasing_sigma=0.5)

        # Recur on the scaled images until they reach the desired pyramid depth
        shift_vector = 2 * np.asarray(imalign_pyramid(scaled_a, scaled_b, level-1))

        # Shift b by the updated shift vector
        b = np.roll(b, shift_vector, axis=(0, 1))
        # Align using a sliding window of size 2 and add the obtained shift vector to the total shift vector
        shift_vector = shift_vector + imalign(a, b, 2)

    return shift_vector