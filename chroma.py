#%% === IMPORTS ===
import sys
sys.path.append("dextr/")
import cv2 as cv
import numpy as np
from scipy import stats

from PIL import Image

from matplotlib import pyplot as plt
from matplotlib import patches as patches

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import dextr.dextr_backend as dextr
import dextr.dataloaders.helpers as helpers
import color_helper.color_helper as c_help
#%% 
def analyze_coral(image_name, rmshad=False, extreme_points=None):
    """ ===== CONSTANTS & PARAMETERS ===== """
    # File Paths
    IMAGE_NAME = image_name
    IMAGE_PATH = 'images/'+IMAGE_NAME+'.jpg'        # Main image of coral
    TEMPLATE_PATH = 'images/KOA_REF.jpg'            # Image of Ko'a Card
    COLOR_REF_PATH = 'data/koa_rgb_refvals.csv'     # CSV of RGB Vals on Ko'a 
    
    # Ko'a Card Constants
    KOA_SHADES = 36                 # Number of shades on Ko'a Card
    KOA_START = 105                 # Location of "0 degs" on card from +x axis. 
    KOA_END = KOA_START + 360       # Full rotation around card
    KOA_INC = 10                    # Arc width of shades on card
    KOA_R = 200                     # Outer radius for color capture
    KOA_RW = 75
    
    # DEXTR Constants
    DEXTR_PAD = 50                  # Padding on bounding box for DEXTR
    DEXTR_THRESH = 0.8              # DEXTR Segmentation threshold
    
    # Homography Constants
    HG_MIN_MATCH_COUNT = 10         # Minimum matches for "good" homography
    HG_TOLERANCE = 2                # Lower tolerance means more selective matches
    
    # Partitioning Constants
    PARTITION_FACTOR = 0.05         # Granularity of partitions
    PARTITION_K = 3                 # Used for k-means neighbors color extraction
    
    # Color Correction Constants
    CC_ON = False
    CC_FEATURES = 3                 # Number of terms for least squares.
    
    # Color Comparison Constants
    COMP_SAMPLES = 10               # Number of times we match coral to shades
    COMP_THRESH = 0.75              # Coral partition weight needed to sample
    COMP_WEDGE = False              # TODO. Wedge color extraction
    COMP_MANUAL = False             # For manual choice of shades
    COMP_AIDED = False              # TODO. User input to help decision thresh.
    COMP_REMOVE_SHADOWS = rmshad    # Remove shadows via Otzu Thresholding. 
    
    """ ===== I/O ===== """
    # Reference Values for Color Correction
    ref_koa_vals = c_help.read_ref_file(COLOR_REF_PATH, KOA_SHADES)
    
    # Main Image
    img = np.array(Image.open(IMAGE_PATH))
    img_bw = cv.imread(IMAGE_PATH,0)
    (img_h, img_w, img_d) = img.shape
    
    # Query or Template Image
    tpl = np.array(Image.open(TEMPLATE_PATH))
    tpl_bw = cv.imread(TEMPLATE_PATH,0)
    (tpl_h, tpl_w, tpl_d) = tpl.shape
    
    # Get DEXTR Bounding Boxes
    plt.close('all')
    if extreme_points is None:
            # Allow pass in of extreme_points to allow same region for 
            # same picture with different analysis parameters
        extreme_points = dextr.get_extreme(img)
    
    # Get representative point if manual selection or user-guided analysis
    rep_point = None
    if COMP_MANUAL or COMP_AIDED:
        rep_point = c_help.get_colorcard_locs(img, 1)
        
    """ ===== Color Card Localization ===== """
    # Color Card Sampling Location Calculation
    # If using point samples instead of wedge method.
    # Find units 360 degrees around. 
    t = np.arange(KOA_START, KOA_END, KOA_INC)
    t = t * (np.math.pi / 180)
    circ_points = np.array(((KOA_R*np.cos(t))+(tpl_w/2), 
                            tpl_h-((KOA_R*np.sin(t))+(tpl_h/2)))).T
    circ_points = np.flip(circ_points, 0).reshape(-1,1,2)
    
    # BRISK + Brute Force Matching
    # Threshold is for FAST, Octaves are for detection at scale
    brisk = cv.BRISK_create(thresh=10, octaves=3)
    # Use crossCheck! Super important
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    # Retrieve keypoints and descriptors
    kp1, des1 = brisk.detectAndCompute(tpl_bw, None)
    kp2, des2 = brisk.detectAndCompute(img_bw, None)
    
    # Find potential matches
    matches = matcher.match(des1, des2)
    
    # Remove terrible matches. Increase tolerance if you want to be more lax.
    distances = [match.distance for match in matches]
    min_dist = min(distances)
    avg_dist = sum(distances) / len(distances)
    min_dist = min_dist or avg_dist * 1.0 / HG_TOLERANCE
    good = [match for match in matches if
                    match.distance <= HG_TOLERANCE * min_dist]
    
    # Find homography with sufficient matches. 
    if len(good)>HG_MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        pts = np.float32([ [0,0],[0,tpl_h-1],
                          [tpl_w-1,tpl_h-1],[tpl_w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        dst_circ = cv.perspectiveTransform(circ_points, M)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 
              HG_MIN_MATCH_COUNT))
        matchesMask = None
        
    """ ===== Color Card Color Retrieval ===== """
    # Select 36 shades of colors
    img_koa_vals = np.zeros((KOA_SHADES, 3))
    if len(good)>HG_MIN_MATCH_COUNT:
        img_koa_locs = np.int32(dst_circ.reshape(36,2))
        for i in range(KOA_SHADES):
            if COMP_WEDGE:
                img_koa_vals[i] = c_help.retrieve_wedge_color(tpl_h, tpl_w, KOA_R,
                                                          i, KOA_INC-1, KOA_SHADES,
                                                          KOA_START, KOA_RW, M, img)
            else:
                img_koa_vals[i] = img[img_koa_locs[i][1], 
                                        img_koa_locs[i][0]]
    else:
        img_koa_locs = c_help.get_colorcard_locs(img, KOA_SHADES,
                                                              caption="Select a point from each of the 36 shades, clockwise from 0 degrees")
        img_koa_vals = img[img_koa_locs[:,0], img_koa_locs[:,1]]
    
    """ ===== Optional: Color Correction ===== """
    # TODO: Does not work very well. Advise leaving off. 
    img_corr = img
    img_koa_vals_corr = img_koa_vals
    if CC_ON:
        # Populate feature matrix
        feat = np.zeros((KOA_SHADES, CC_FEATURES))
        feat[:, 0] = img_koa_vals[:,0]
        feat[:, 1] = img_koa_vals[:,1]
        feat[:, 2] = img_koa_vals[:,2]
       
        # Solve for Coefficients
        (r_coeff, _, _, _) = np.linalg.lstsq(feat, ref_koa_vals[:,0], rcond=None)
        (g_coeff, _, _, _) = np.linalg.lstsq(feat, ref_koa_vals[:,1], rcond=None)
        (b_coeff, _, _, _) = np.linalg.lstsq(feat, ref_koa_vals[:,2], rcond=None)
        coeff = np.vstack((r_coeff, g_coeff, b_coeff))
        
        img_corr = img.astype(float)
        for x in range(img_w):
            for y in range(img_h):
                pixel = img[y, x].astype(float)
                pixel_feat = np.zeros(CC_FEATURES, dtype=np.float64)
                pixel_feat[0] = pixel[0]
                pixel_feat[1] = pixel[1]
                pixel_feat[2] = pixel[2]
                img_corr[y, x] = np.dot(coeff,pixel_feat)
        
        img_corr = np.clip(img_corr, 0, 255)
        img_corr = img_corr.astype(np.uint8)
        
        # Re-Retrieve 36 Colors
        for i in range(KOA_SHADES):
            img_koa_vals_corr[i] = img_corr[img_koa_locs[i][1], 
                                                    img_koa_locs[i][0]]
        
    """ ===== Segmentation ===== """
    # Get Extreme Points
    mask, bbox_minmax = dextr.get_mask(img, extreme_points, 
                                       DEXTR_PAD, DEXTR_THRESH)
    mask_arr = mask[0]

    # Get Bbox Tuples
    bbox_top = bbox_minmax[1]
    bbox_bot = bbox_minmax[3]
    bbox_lef = bbox_minmax[0]
    bbox_rig = bbox_minmax[2]
    
    topx, topy = [bbox_lef, bbox_rig], [bbox_top, bbox_top]
    botx, boty = [bbox_lef, bbox_rig], [bbox_bot, bbox_bot]
    lefx, lefy = [bbox_lef, bbox_lef], [bbox_top, bbox_bot]
    rigx, rigy = [bbox_rig, bbox_rig], [bbox_top, bbox_bot]
    
    # === Partitioning ===
    # Get Width and Height
    bbox_wid = bbox_rig - bbox_lef
    bbox_hei = bbox_bot - bbox_top
    partition_wid = np.int(bbox_wid * PARTITION_FACTOR)
    partition_hei = np.int(bbox_hei * PARTITION_FACTOR)
    partition_area = partition_wid * partition_hei
    
    partition_x = np.arange(bbox_lef, bbox_rig, partition_wid)
    partition_y = np.arange(bbox_top, bbox_bot, partition_hei)
    
    partition_x_matrix = partition_x[:, np.newaxis]
    partition_x_matrix = np.repeat(partition_x_matrix, partition_y.size, 1)
    partition_y_matrix = partition_y[:, np.newaxis]
    partition_y_matrix = np.repeat(partition_y_matrix, partition_x.size, 1)
    
    # === Partition Color Extraction ===
    # Find dominant color in each square
    color_matrix = np.zeros((partition_x.size-1, partition_y.size-1, 3))
    weight_matrix = np.zeros((partition_x.size-1, partition_y.size-1))
    for x in range(partition_x.size-1):
        for y in range(partition_y.size-1):
            x1 = partition_x[x]
            y1 = partition_y[y]
            x2 = partition_x[x+1]
            y2 = partition_y[y+1]
            # Remeber that array indexing is row-major, so starts with y indices
            partition = img_corr[y1:y2, x1:x2, :]
            color = c_help.get_dominant_color(partition, k=PARTITION_K, 
                                                     remove_shadows=COMP_REMOVE_SHADOWS)
            color_matrix[x,y] = color
            weight_matrix[x,y] = np.sum(mask_arr[y1:y2, x1:x2]) / partition_area
    
    # === Dominant Color Extraction === 
    dominant_colors = []
    if COMP_MANUAL:
        px,py = rep_point[0][0], rep_point[0][1]
        partitiond = img_corr[py-10:py+11, px-10:px+11, :]
        dominant_colors.append(c_help.get_dominant_color
                               (partitiond, k=PARTITION_K, 
                                remove_shadows=COMP_REMOVE_SHADOWS))
    else:
        valid_partitions = weight_matrix > COMP_THRESH
        x, y = np.indices((weight_matrix.shape[0], weight_matrix.shape[1]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T.reshape(weight_matrix.shape[0],weight_matrix.shape[1],2)
        valid_points =  np.float32(points[valid_partitions]).reshape(-1,2).astype(int)
        values = valid_points[(np.random.rand(COMP_SAMPLES)
                                *valid_points.shape[0]).astype(int)]
        dominant_colors = np.asarray(color_matrix[values[:,0], values[:,1]])
    
    # === Color Match to 36 Shades ===
    match_indices = []
    match_colors = []
    match_diffs = []
    img_colorcard_lab = []
    for color in img_koa_vals_corr:
        color_rgb = sRGBColor(color[0], color[1], color[2])
        color_lab = convert_color(color_rgb, LabColor);
        img_colorcard_lab.append(color_lab)
        
    for dominant_color in dominant_colors:
        dominant_color_rgb = sRGBColor(dominant_color[0],
                                       dominant_color[1], dominant_color[2])
        dominant_color_lab = convert_color(dominant_color_rgb, LabColor);
        
        color_diffs = np.zeros(KOA_SHADES)
        for i in range(len(img_colorcard_lab)):
            color = img_colorcard_lab[i]
            delta_e = delta_e_cie2000(color, dominant_color_lab);
            color_diffs[i] = delta_e
        
        match_index = np.argmin(color_diffs)
        match_color = img_koa_vals_corr[match_index]
        match_diff = color_diffs[match_index]
        
        match_indices.append(match_index)
        match_colors.append(match_color)
        match_diffs.append(match_diff)
    
    #=== Plot Everything ===
    # Plot the results
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    plt.axis('off')
    
    # Original Image
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Homography
    ax2.imshow(img)
    if len(good)>HG_MIN_MATCH_COUNT:
        img_bw = cv.polylines(img_bw,[np.int32(dst)],True,255,3, cv.LINE_AA)
        img_bw = cv.polylines(img_bw,[np.int32(dst_circ)],True,255,3, cv.LINE_AA)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
        img3 = cv.drawMatches(tpl_bw,kp1,img_bw,kp2,good,None,**draw_params)
        ax2.imshow(img3, 'gray')
    ax2.set_title('Homography')
    ax2.axis('off')
    
    # Segmented
    ax3.imshow(img_corr)
    ax3.imshow(helpers.overlay_masks(img_corr / 256, mask))
    if rep_point is not None:
        ax3.plot(rep_point[0][0], rep_point[0][1], 'gx')
    ax3.plot(extreme_points[:, 0], extreme_points[:, 1], 'bx')
    ax3.plot(topx, topy, botx, boty, lefx, lefy, rigx, rigy, 
             marker = 'o', c='black', linestyle='solid')
    ax3.set_title('Segmented')
    ax3.axis('off')
    
    # Masked
    ax4.imshow(img_corr)
    ax4.imshow(helpers.overlay_masks(img_corr / 256, mask))
    if rep_point is not None:
        ax4.plot(rep_point[0][0], rep_point[0][1], 'gx')
    ax4.plot(extreme_points[:, 0], extreme_points[:, 1], 'bx')
    ax4.plot(topx, topy, botx, boty, lefx, lefy, rigx, rigy, 
             marker = 'o', c='black', linestyle='solid')
    ax4.set_xlim(partition_x[0], partition_x[-1])
    ax4.set_ylim(partition_y[0], partition_y[-1])
    ax4.set_title('Segmented (Magnified)')
    ax4.axis('off')
    
    # Partitions
    ax5.imshow(img_corr)
    for row in partition_x_matrix:
        ax5.plot(row, partition_y, c='black', linestyle='dashed')
    for col in partition_y_matrix:
        ax5.plot(partition_x, col, c='black', linestyle='dashed')
    for x in range(partition_x.size-1):
        for y in range(partition_y.size-1):
            x1 = partition_x[x]
            y1 = partition_y[y]
            x2 = partition_x[x+1]
            y2 = partition_y[y+1]
            color = color_matrix[x,y]
            color = np.clip(color, 0, 255)
            rect = patches.Rectangle((x1,y1),partition_wid,partition_hei,
                                     linewidth=1,edgecolor=color/256,
                                     facecolor=color/256)
            ax5.add_patch(rect)
    ax5.set_xlim(partition_x[0]-50, partition_x[-1]+51)
    ax5.set_ylim(partition_y[0]-50, partition_y[-1]+51)
    ax5.set_title('Partitioned')
    ax5.axis('off')
    
    # Dominant Color Output
    ax6.imshow(img_corr)
    modal_index = int(stats.mode(match_indices)[0])
    modal_color = dominant_colors[match_indices.index(modal_index)]
    modal_diff = match_diffs[match_indices.index(modal_index)]
    
    deg_str = "{} degrees".format(modal_index*10)
    dif_str = " Delta-E {:03.2f}".format(modal_diff)
    
    # Plot Candidate Colors
    w_inc = img_w / 4
    h_inc = img_h / len(dominant_colors) 
    for i in range(len(dominant_colors)):
        color = dominant_colors[i]
        rect = patches.Rectangle((0,h_inc*i), w_inc, h_inc,
                                 linewidth=1,edgecolor=[0,0,0],
                                 facecolor=color/256)
        ax6.add_patch(rect)
    
    # Plot Modal Color
    rect = patches.Rectangle((w_inc,0), w_inc, img_h,
                             linewidth=1,edgecolor=[0,0,0],
                             facecolor=modal_color/256)
    ax6.add_patch(rect)
    
    # Plot In-Scene Match
    rect = patches.Rectangle((w_inc*2,0), w_inc,img_h,
                             linewidth=1,
                             edgecolor=[0,0,0],
                             facecolor=img_koa_vals_corr[modal_index]/256)
    ax6.add_patch(rect)
    
    # Plot Reference Match
    rect = patches.Rectangle((w_inc*3,0), w_inc,img_h,
                             linewidth=1,
                             edgecolor=[0,0,0],
                             facecolor=ref_koa_vals[modal_index]/256)
    ax6.add_patch(rect)
    ax6.text(w_inc*3, h_inc, deg_str, fontsize=6)
    ax6.text(w_inc*3, h_inc*2, dif_str, fontsize=6)
    ax6.set_title("Candidates | Most Common | In-Scene Match | Reference Value", 
                  fontsize=7)
    ax6.axis('off')
    
    fig.set_size_inches(12,6)
    if COMP_REMOVE_SHADOWS:
        fig.savefig('output/'+IMAGE_NAME+'S.png', bbox_inches='tight', dpi=1000)
    else:   
        fig.savefig('output/'+IMAGE_NAME+'.png', bbox_inches='tight', dpi=1000)
        
    plt.close('all')
    return extreme_points
#%%
images = ['S3E1_4']
for img in images:
    ext = analyze_coral(img, False)
    analyze_coral(img, True, ext)
#%%
ratings = c_help.read_ref_file('data/shadcorr.csv', 84)
i = 0
fig,ax = plt.subplots(1)
x = np.linspace(0,750,13)
x = x + x[1]/2
x = x[0:len(x)-1]
xticks = ['1', '2', '3', '4', '5', '6', 
          '7', '8', '9', '10', '11', '12']
ax.set_xlabel('Specimen')
y = np.linspace(0,400,8)
y = y + y[1]/2
y = y[0:len(x)-1]
yticks = ['MV', 'ML', 'LS', 'JRO', 'EM', 'Computer', 'Shadow Corrected']
plt.xticks(x, xticks)
plt.yticks(y, yticks)
i=0
w=750
h=400
w_inc = w / 12
h_inc = h / 7
bg = np.zeros((400,750,3))
ax.imshow(bg)
for color in ratings:
    rect = patches.Rectangle((w_inc*(i//7),h_inc*(i%7)), w_inc,h_inc,
                             linewidth=1,
                             edgecolor=[0,0,0],
                             facecolor=color/256)
    ax.add_patch(rect)
    i += 1
ax.set_title("Human Observer + Computer + Shadow Corrected Color Ratings")
fig.savefig('output/shadcorr.png', bbox_inches='tight', dpi=1000)