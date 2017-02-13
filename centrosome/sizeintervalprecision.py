import _sizeIntervalPrecision as sip
import numpy as np
import math
   
def sizeintervalprecision(data, min_diameter, max_diameter, ignore_large=False):
    """<b>SizeIntervalPrecision</b> Object Feature based Gray-level Threshold using optimized global precision
    <hr>
    <b>SizeIntervalPrecision</b> Computes the threshold level that optimizes the global precision with regards to a given
    expected object size range.

    by Petter Ranefall 2016

    "Global Gray-level Thresholding Based on Object Size", Cytometry Part A, 89:4, 2016, pp. 385-390.

    data           - an array of intensity values between zero and one
    min_diameter   - minimum value of typical diameter of objects, in pixel units
    max_diameter   - maximum value of typical diameter of objects, in pixel units
    ignore_large   - use this setting to ignore objects above the upper interval limit. This is used to handle clustered objects. The recommendation is that this should be set to True for most applications where there is a risk of having clustered objects.
    """
    assert data.ndim == 2
    nPix = np.prod(data.shape)
    if nPix == 0:
        return 0
    elif nPix == 1:
        return data[0][0]
        
    r_min = min_diameter / 2.0
    r_max = max_diameter / 2.0
    area_interval_min = (int)(math.pi * r_min * r_min)
    area_interval_max = (int)(math.pi * r_max * r_max)
    thr = sip._sizeIntervalPrecision(data, (area_interval_min, area_interval_max), (0, 1), int(ignore_large))
    
    return thr
