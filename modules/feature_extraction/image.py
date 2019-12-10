from itertools import product
import numbers

import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.feature_extraction.image import _extract_patches


def _compute_n_patches(i_h, i_w, p_h, p_w, e_h, e_w, max_patches=None):
    """
    Compute the number of patches that will be extracted in an image.

    Parameters
    ----------
    i_h : int
        The image height

    i_w : int
        The image with

    p_h : int
        The height of a patch

    p_w : int
        The width of a patch

    e_h : int
        The vertical extraction step

    e_w : int
        The horizontal extraction step

    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
    n_h = (i_h - p_h) // e_h + 1
    n_w = (i_w - p_w) // e_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Integral))
              and max_patches >= all_patches):
            return all_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def extract_patches_2d(img, patch_size, extraction_step=1, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches
    The resulting patches are allocated in a dedicated array.

    Parameters
    ----------
    img : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    extraction_step : integer or tuple of length arr.ndim
            Indicates step size at which extraction shall be performed.
            If integer is given, then the step is uniform in all dimensions.

    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int, RandomState instance or None, optional (default=None)
        Determines the random number generator used for random sampling when
        `max_patches` is not None. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.
    """
    i_h, i_w = img.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    if isinstance(extraction_step, numbers.Number):
        e_h, e_w = extraction_step, extraction_step
    else:
        e_h, e_w = extraction_step

    img = check_array(img, allow_nd=True)
    img = img.reshape((i_h, i_w, -1))
    n_colors = img.shape[-1]

    extracted_patches = _extract_patches(img,
                                         patch_shape=(p_h, p_w, n_colors),
                                         extraction_step=(e_h, e_w, n_colors))

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, e_h, e_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches


def reconstruct_from_patches_2d(patches, image_size, extraction_step=1):
    """
    Reconstruct the image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.

    image_size : tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed

    extraction_step : integer or tuple of length arr.ndim
            Indicates step size at which extraction shall be performed.
            If integer is given, then the step is uniform in all dimensions.

    Returns
    ----------
    img : array, shape = image_size
        the reconstructed image
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]

    if isinstance(extraction_step, numbers.Number):
        e_h, e_w = extraction_step, extraction_step
    else:
        e_h, e_w = extraction_step

    img = np.zeros(image_size)
    counter = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = (i_h - p_h) // e_h + 1
    n_w = (i_w - p_w) // e_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i*e_h:i*e_h + p_h, j*e_w:j*e_w + p_w] += p
        counter[i*e_h:i*e_h + p_h, j*e_w:j*e_w + p_w] += 1

    counter[counter == 0] = 1

    return img / counter
