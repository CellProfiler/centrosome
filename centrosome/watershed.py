import skimage.morphology


def watershed(image, markers, connectivity=None, offset=None, mask=None):
    skimage.morphology.watershed(image, markers, connectivity, offset, mask)
