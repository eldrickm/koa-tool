# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import numpy as np
import csv
import cv2 #for resizing image

def get_dominant_color(image, k=4, image_processing_size = None, remove_shadows = False):
    """
    Credit to:
    https://adamspannbauer.github.io/2018/03/02/app-icon-dominant-colors/#plot
    
    takes an image as input
    returns the dominant color of the image as a list
    
    dominant color is found by running k means on the 
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image; 
    this resizing can be done with the image_processing_size param 
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    # Resize image if new size provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
        
    if remove_shadows:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image = image[(th>0)]
    else:
        # Reshape image to two dimensions
        image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    #if remove_shadows:
        # ITU BT.601:
        #Y = 0.299 R + 0.587 G + 0.114 B
        #image.tolist().sort(key=lambda x: 0.299 * x[0] + 0.587* x[1] + 0.114* x[2])
        #image = image[len(image)//2:]
    
    # Cluster and fit pixels
    if len(image) < k:
        return np.mean(image)
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    # Find most popular label
    label_counts = Counter(labels)

    # Return cluster center with most neighbors
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return dominant_color

def get_colorcard_locs(image, num_locs, caption="Select A Representative Point for Analysis"):
    #  read image and click the points
    plt.ion()
    plt.axis('off')
    plt.imshow(image)
    plt.title(caption)
    colorcard_locs = np.array(plt.ginput(num_locs, timeout=0)).astype(np.int)
    plt.close()
    return colorcard_locs

def retrieve_wedge_color(tplh, tplw, radius, idx, inc, num_shades, origin, r_width, M, img):
    """
    # TODO: Wedge retrieval for more robust color retrieval. 
    koa_fig, koa_ax = plt.subplots(1, 1)
    koa_ax.imshow(tpl)
    wedge = patches.Wedge((250,250), 180, 90, 100, width=30)
    koa_ax.add_patch(wedge)
    
    x, y = np.indices((500,500))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    
    p = wedge.get_path()
    grid = p.contains_points(points)
    grid = grid.reshape(500,500) 
    
    points = points.reshape(500,500,2)
    test =  np.float32(points[grid]).reshape(-1,1,2)
    
    dst_test = cv.perspectiveTransform(test,M)
    dst_test = dst_test.astype(int)
    dst_arr = np.asarray(dst_test)
    
    average_red = np.average(img[dst_arr[:,:,1],dst_arr[:,:,0],0])
    average_green= np.average(img[dst_arr[:,:,1],dst_arr[:,:,0],1])
    average_blue = np.average(img[dst_arr[:,:,1],dst_arr[:,:,0],2])
    """
    koa_fig, koa_ax = plt.subplots(1, 1)
    theta1 = idx*inc+origin
    theta2 = theta1 + inc
    wedge = patches.Wedge((tplh//2, tplw//2), radius, theta1, 
                          theta2, width=r_width)
    x, y = np.indices((500,500))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    
    p = wedge.get_path()
    grid = p.contains_points(points)
    grid = grid.reshape(500,500) 

    points = points.reshape(500,500,2)
    test =  np.float32(points[grid]).reshape(-1,1,2)

    dst_test = cv2.perspectiveTransform(test,M)
    dst_test = dst_test.astype(int)
    dst_arr = np.asarray(dst_test)

    average_red = np.average(img[dst_arr[:,:,1],dst_arr[:,:,0],0])
    average_green= np.average(img[dst_arr[:,:,1],dst_arr[:,:,0],1])
    average_blue = np.average(img[dst_arr[:,:,1],dst_arr[:,:,0],2])
    return average_red, average_green, average_blue

def read_ref_file(filename, num_shades):
    ref_vals = np.zeros((num_shades, 3))
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if (line_count > 0) & (line_count <= num_shades):
                ref_vals[line_count-1] = row[1:]
            line_count += 1
    return ref_vals