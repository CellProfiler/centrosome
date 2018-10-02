/* _sizeIntervalPrecision.c - C routines for SizeIntervalPrecision thresholding
 *
 * CellProfiler is distributed under the GNU General Public License,
 * but this file is licensed under the more permissive BSD license.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
 * All rights reserved.
 *
 * Please see the AUTHORS file for credits.
 *
 * Website: http://www.cellprofiler.org
 *
 *
 * sizeIntervalPrecision - compute threshold
 *             the context of a structuring element.
 *             arg1 - the float64 ndarray image
 *             arg2 - size interval (a 2-tuple)
 *             arg3 - threshold constraint (a 2-tuple)
 */
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>

static void to_stdout(const char *text)
{
     PySys_WriteStdout("%s", text);     
}

struct pair {
    double value;
    long index;
};

long my_pair_compare(const void *const first, const void *const second)
{
    const struct pair* a;
	const struct pair* b;
	a = (const struct pair*)first;
    b = (const struct pair*)second;
    if (a->value > b->value)
       return 1;
    if (a->value < b->value)
        return -1;
    if (a->index > b->index)
       return 1;
    return -1;
}

int sortIndices(long count, double values[], long sortedIndices[])
{
    struct pair* ab;
	long i;
	ab = (struct pair *) malloc(sizeof(struct pair)*count);
	
	if (!ab)
	{
		return 0;
	}
	
    for (i = 0; i<count; ++i) {
        ab[i].value = values[i];
        ab[i].index = i;
    }
    qsort(ab, count, sizeof(*ab), my_pair_compare);
    for (i=0; i<count; ++i){
        sortedIndices[i] = ab[i].index;
    }
	free(ab);
	
	return 1;
}

long _findNode(long n, long parentNodes[])
{
	long root;
    if (parentNodes[n] != n)
	{
        root = _findNode(parentNodes[n], parentNodes);
        parentNodes[n] = root;
        return root;
	}
    return n;
}

long _mergeNodes(long n1, long n2, long areaLowerBound, long areaUpperBound, long lenData, 
				long intData[], long parentNodes[], long accumulatedSizes[], long pixelCountInsideIntervalHist[], long pixelCountOutsideIntervalHist[], long ignoreLarge)
{
	long par;
	long child;
	long i;
	long objectSize;
	
    if (intData[n1] == intData[n2])
	{
        par = max(n1, n2);
        child = min(n1, n2);
        parentNodes[child] = par;
        accumulatedSizes[par] += accumulatedSizes[child];
	}
    else // pixels[n1] is always > pixels[n2] here
	{
        par = n2;
        child = n1;
        parentNodes[child] = par;
        accumulatedSizes[par] += accumulatedSizes[child];
        objectSize = accumulatedSizes[child];
        if (objectSize >= areaLowerBound && objectSize <= areaUpperBound)
		{
            for (i = intData[child]; i > intData[par]; --i) //Decreasing level from child downto the level above parent
                pixelCountInsideIntervalHist[i] += objectSize;
		}
        else if (objectSize < areaLowerBound || (objectSize > areaUpperBound && !ignoreLarge))
		{
            for (i = intData[child]; i > intData[par]; --i) //Decreasing level from child downto the level above parent
                pixelCountOutsideIntervalHist[i] += objectSize;
		}
	}
    return par;
}

double _constrain(double threshold, double minThr, double maxThr)
{
	if (threshold < minThr)
		return minThr;
	if (threshold > maxThr)
		return maxThr;
	return threshold;
}

static PyObject *
_sizeIntervalPrecision(PyObject *self, PyObject *args)
{
	PyObject *image     = NULL; /* the image ndarray */
	PyObject *cimage    = NULL; /* a contiguous version of the array */
	PyObject *im_shape  = NULL; /* the image shape sequence */
    PyObject *sizeInterval = NULL; /* 2-tuple giving the size interval */
    PyObject *thrInterval = NULL; /* 2-tuple giving the size interval */
    PyObject *ignoreLargeObj = NULL; /* double giving the ellipse fit threshold */
	long     height     = 0;    /* the image height */
	long     width      = 0;    /* the image width */
	double   *imdata    = NULL; /* the image data */
	const char *error   = NULL;
	PyObject **shapes[] = { &im_shape, &im_shape };
	long     *slots[]   = { &height, &width };
    long      indices[]  = { 0,1 };
	long      i          = 0;
	long      j          = 0;
	long      k          = 0;
	long      l          = 0;
	npy_intp dims[2];
	double 	 res = 1.0;
	long lenData;
	long *sortedIndices = NULL;
	long *intData = NULL;
	long curLevel;
	long nUnique;
	long *intHist = NULL;
	long *cumIntHist = NULL;
	
    long *parentNodes = NULL;
    long *accumulatedSizes = NULL;
    long *pixelCountInsideIntervalHist = NULL;
    long *pixelCountOutsideIntervalHist = NULL;
	long x;
	long y;
	long prevX;
	long prevY;
	long nextX;
	long nextY;
	long curNode;
	long adjNode;
	long areaLowerBound;
	long areaUpperBound;
	double thrLowerBound;
	double thrUpperBound;
	long ignoreLarge;
	double *precision = NULL;
	double *recall = NULL;
    long maxPixelCountInside;
    double maxPrecision;
    long thr;
	
	image = PySequence_GetItem(args,0);
	if (! image) {
	  error = "Failed to get image from arguments";
	  goto exit;
	}
	cimage = PyArray_ContiguousFromAny(image, NPY_DOUBLE,2,2);
	if (! cimage) {
	  error = "Failed to make a contiguous array from first argument";
	  goto exit;
	}
	im_shape = PyObject_GetAttrString(cimage,"shape");
	if (! im_shape) {
	  error = "Failed to get image.shape";
	  goto exit;
	}
	if (! PyTuple_Check(im_shape)) {
	  error = "image.shape not a tuple";
	  goto exit;
	}
	if (PyTuple_Size(im_shape) != 2) {
	  error = "image is not 2D";
	  goto exit;
	}
	for (i=0;i<2;i++) {
		PyObject *obDim = PyTuple_GetItem(*shapes[i],indices[i]);
		*(slots[i]) = PyInt_AsLong(obDim);
		if (PyErr_Occurred()) {
			error = "Array shape is not a tuple of integers";
			goto exit;
		}
	}
	
	dims[0] = width;
	dims[1] = height;	
	
	sizeInterval = PySequence_GetItem(args,1);
	if (! sizeInterval) 
	{
		  error = "Failed to get sizeInterval into structure from args";
		  goto exit;
	}
	if (! PyTuple_Check(sizeInterval)) 
	{
		  error = "sizeInterval is not a tuple";
		  goto exit;
	} 
	else 
	{
		PyObject *temp = PyTuple_GetItem(sizeInterval,0);
		if (! temp) 
		{
		   error = "Failed to get areaLowerBound from tuple";
		   goto exit;
		}
		areaLowerBound = PyInt_AsLong(temp);
		if (PyErr_Occurred()) 
		{
		   error = "areaLowerBound is not an integer";
		   goto exit;
		}
		temp = PyTuple_GetItem(sizeInterval,1);
		if (! temp) 
		{
		   error = "Failed to get areaUpperBound from tuple";
		   goto exit;
		}
		areaUpperBound = PyInt_AsLong(temp);
		if (PyErr_Occurred()) 
		{
		   error = "areaUpperBound is not an integer";
		   goto exit;
		}
	}
	
	thrInterval = PySequence_GetItem(args,2);
	if (! thrInterval) 
	{
		  error = "Failed to get thrInterval into structure from args";
		  goto exit;
	}
	if (! PyTuple_Check(thrInterval)) 
	{
		  error = "thrInterval is not a tuple";
		  goto exit;
	} 
	else 
	{
		PyObject *temp = PyTuple_GetItem(thrInterval,0);
		if (! temp) 
		{
		   error = "Failed to get thrLowerBound from tuple";
		   goto exit;
		}
		thrLowerBound = PyFloat_AsDouble(temp);
		if (PyErr_Occurred()) 
		{
		   error = "thrLowerBound is not a double";
		   goto exit;
		}
		temp = PyTuple_GetItem(thrInterval,1);
		if (! temp) 
		{
		   error = "Failed to get thrUpperBound from tuple";
		   goto exit;
		}
		thrUpperBound = PyFloat_AsDouble(temp);
		if (PyErr_Occurred()) 
		{
		   error = "thrUpperBound is not an integer";
		   goto exit;
		}
	}
	
	ignoreLargeObj = PySequence_GetItem(args,3);
	if (! ignoreLargeObj) 
	{
		  error = "Failed to get ignoreLargeObj into structure from args";
		  goto exit;
	}
	ignoreLarge = PyInt_AsLong(ignoreLargeObj);
	
	res = thrUpperBound;
	
	lenData = width * height;
	
	sortedIndices = (long *) malloc(sizeof(long)*lenData);
	if (!sortedIndices)
	{
		error = "Failed to allocate memory";
		goto exit;
	}
	
	imdata = (double *)PyArray_DATA(cimage);
	if (! imdata) {
		error = "Failed to get image data";
		goto exit;
	}
	
	if (!sortIndices(lenData, imdata, sortedIndices))
	{
		error = "Failed to allocate memory";
		goto exit;
	}
	intData = (long *) malloc(sizeof(long)*lenData);
	if (!intData)
	{
		error = "Failed to allocate memory";
		goto exit;
	}
    curLevel = 0;
	intData[sortedIndices[0]] = curLevel;
    for(i = 1; i < lenData; ++i)
	{
        if(imdata[sortedIndices[i]] != imdata[sortedIndices[i-1]])
            curLevel++;
        intData[sortedIndices[i]] = curLevel;
	}
	
	//Allocate arrays
	nUnique = intData[sortedIndices[lenData-1]] + 1;
    intHist = (long *) calloc(nUnique, sizeof(long));
	cumIntHist = (long *) malloc(sizeof(long)*nUnique);
    pixelCountInsideIntervalHist = (long *) calloc(nUnique, sizeof(long));
    pixelCountOutsideIntervalHist = (long *) calloc(nUnique, sizeof(long));	
    precision = (double *) malloc(sizeof(double)*nUnique); 
    recall = (double *) malloc(sizeof(double)*nUnique);	
    parentNodes = (long *) malloc(sizeof(long)*lenData); 
    accumulatedSizes = (long *) malloc(sizeof(long)*lenData);
	if (!(intHist && cumIntHist && pixelCountInsideIntervalHist && pixelCountOutsideIntervalHist && 
		precision && recall && parentNodes && accumulatedSizes))
	{
		error = "Failed to allocate memory";
		goto exit;
	}
	
    for(i = 0; i < lenData; ++i)
	{
		intHist[intData[i]] ++;
	}
	cumIntHist[0] = intHist[0];
	for(i=1; i < nUnique; ++i)
	{
        cumIntHist[i] = intHist[i] + cumIntHist[i-1];
	}

    for(i = 0; i < lenData; ++i)
	{
		parentNodes[i] = i;
		accumulatedSizes[i] = 1;
	}
    for(i = lenData-1; i >= 0; i--)
	{
        j = sortedIndices[i];
        curNode = j;
        y = j/width;
        x = j-y*width;
        prevY = y-1;
        nextY = y+1;
        prevX = x-1;
        nextX = x+1;

        if (nextY<height)
		{
            k = x+width*nextY;
            if (intData[k] >= intData[j])
			{
                adjNode = _findNode(k, parentNodes);
                if (curNode != adjNode)
                    curNode = _mergeNodes(adjNode, curNode, areaLowerBound, areaUpperBound, lenData, intData, parentNodes, accumulatedSizes, pixelCountInsideIntervalHist, pixelCountOutsideIntervalHist, ignoreLarge);
			}
		}
        if (nextX<width)
		{
            k = nextX+width*y;
            if (intData[k] >= intData[j])
			{
                adjNode = _findNode(k, parentNodes);
                if (curNode != adjNode)
                    curNode = _mergeNodes(adjNode, curNode, areaLowerBound, areaUpperBound, lenData, intData, parentNodes, accumulatedSizes, pixelCountInsideIntervalHist, pixelCountOutsideIntervalHist, ignoreLarge);
			}
		}
        if (prevX >= 0)
		{
            k = prevX+width*y;
            if (intData[k] > intData[j])
			{
                adjNode = _findNode(k, parentNodes);
                if (curNode != adjNode)
                    curNode = _mergeNodes(adjNode, curNode, areaLowerBound, areaUpperBound, lenData, intData, parentNodes, accumulatedSizes, pixelCountInsideIntervalHist, pixelCountOutsideIntervalHist, ignoreLarge);
			}
		}
        if (prevY >= 0)
		{
            k = x+width*prevY;
            if (intData[k] > intData[j])
			{
                adjNode = _findNode(k, parentNodes);
                if (curNode != adjNode)
                    curNode = _mergeNodes(adjNode, curNode, areaLowerBound, areaUpperBound, lenData, intData, parentNodes, accumulatedSizes, pixelCountInsideIntervalHist, pixelCountOutsideIntervalHist, ignoreLarge);
			}
		}
    }
	
    maxPixelCountInside = 0;
    //Compute maximum pixel count inside the interval === Maximum #true positive pixels, given that #positive pixels > #negative pixels
    for (i = 0; i < nUnique; ++i)
	{
		if (pixelCountInsideIntervalHist[i] > maxPixelCountInside && pixelCountInsideIntervalHist[i] > pixelCountOutsideIntervalHist[i])
            maxPixelCountInside = pixelCountInsideIntervalHist[i];
	}
    if (maxPixelCountInside == 0)
	{
        res = thrUpperBound;
		goto exit;
	}
    
    //Compute precision and recall values
    for (i = 0; i < nUnique; ++i)
	{
        if (pixelCountInsideIntervalHist[i] > 0)
		{
            precision[i] = pixelCountInsideIntervalHist[i] / (double)(pixelCountInsideIntervalHist[i] + pixelCountOutsideIntervalHist[i]);
            recall[i] = pixelCountInsideIntervalHist[i] / (double)(maxPixelCountInside);
		}
        else
		{
            precision[i] = 0;
            recall[i] = 0;
		}
	}
    //Compute threshold level as the level that gives the highest precision, given that recall is > 0.5
    maxPrecision = 0.5;
    thr = 0;
    for (i = 0; i < nUnique; ++i)
	{
        if (precision[i] > maxPrecision && recall[i] > 0.5)
		{
            maxPrecision = precision[i];
            thr = i;
		}
	}
    
    if (maxPrecision == 0.5) //No positive value found
	{
        res = thrUpperBound;
	}
	else
	{
		res = imdata[sortedIndices[cumIntHist[thr]-1]];
		res = _constrain(res, thrLowerBound, thrUpperBound);
	}

exit:
	if (image) {
		Py_DECREF(image);
	}
	if (cimage) {
		Py_DECREF(cimage);
	}
	if (im_shape) {
		Py_DECREF(im_shape);
	}
	if (sizeInterval) {
		Py_DECREF(sizeInterval);
	}
	if (thrInterval) {
		Py_DECREF(thrInterval);
	}

	if (error) {
		to_stdout(error);
	}
	
	free(sortedIndices);
	free(intData);
	free(intHist);
	free(cumIntHist);
	free(parentNodes); 
	free(accumulatedSizes);
	free(pixelCountInsideIntervalHist);
	free(pixelCountOutsideIntervalHist);
	
	return Py_BuildValue("f", res);
}

static PyMethodDef methods[] = {
    {"_sizeIntervalPrecision", (PyCFunction)_sizeIntervalPrecision, METH_VARARGS, NULL},
    { NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_sizeIntervalPrecision(void)
{
    Py_InitModule("_sizeIntervalPrecision", methods);
    import_array();
}

