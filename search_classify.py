import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from common_functions import *
from lane_functions import *
import pickle
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations #18
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 720] # Min and max in y to search in slide_window()


def load_data():    
# Read in cars and notcars
    cars = glob.glob('vehicles/*/*.png')
    notcars = glob.glob('non-vehicles/*/*.png')
    # sample_size = 500
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]
    car_features = extract_features(cars, color_space=color_space, 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                                       spatial_size=spatial_size, hist_bins=hist_bins, 
                                       orient=orient, pix_per_cell=pix_per_cell, 
                                       cell_per_block=cell_per_block, 
                                       hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)  
    print ("shape of car_features",len(car_features))
    print ("shape of notcar_features",len(notcar_features))

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))    
    return scaled_X,y,X_scaler


def train_model():
    X,y,X_scaler=load_data()
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
          'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    filename = 'car_detector_svc_v2.pkl'
    pickle.dump(svc, open(filename, 'wb'))

    filename = 'standard_scaler_v2.pkl'
    pickle.dump(X_scaler, open(filename, 'wb'))

    return svc

def process_image(image,classifier,X_scaler,dist_pickle):
    """ Assume RGB image. """
    draw_image = np.copy(image)
    hot_windows=find_cars(draw_image,
                          ystart=y_start_stop[0],
                          ystop=y_start_stop[1],
                          scale=1.5,
                          svc=classifier,
                          X_scaler=X_scaler,
                          orient=orient,
                          pix_per_cell=pix_per_cell,
                          cell_per_block=cell_per_block,
                          spatial_size=spatial_size,
                          hist_bins=hist_bins)

    pruned_windows=prune_false_positives(draw_image,hot_windows)
    lane_image=find_lane(draw_image,dist_pickle)
    pruned_img=draw_boxes(lane_image,pruned_windows)
    return pruned_img


def test_on_image(filename):
# Test on Single Images
    classifier = pickle.load(open("car_detector_svc_v2.pkl", 'rb'))
    X_scaler = pickle.load(open("standard_scaler_v2.pkl", 'rb'))
    dist_pickle = pickle.load(open("camera.pkl", 'rb'))
    image=cv2.imread(filename)[...,::-1]#flip BGR to RGB
    t=time.time()
    pruned_img=process_image(image,classifier,X_scaler,dist_pickle)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to process one image ...')
    
    plt.imshow(pruned_img)
    plt.savefig("output_images/test1_pruned_windows_final.png")

# Test on Video

def test_on_video(videofile):
    clear_lines()
    classifier = pickle.load(open("car_detector_svc_v2.pkl", 'rb'))
    X_scaler = pickle.load(open("standard_scaler_v2.pkl", 'rb'))
    dist_pickle = pickle.load(open("camera.pkl", 'rb'))
    custom_process_image=lambda image: process_image(image,classifier,X_scaler,dist_pickle)
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML
    white_output = videofile+'_output.mp4'
    clip1 = VideoFileClip(videofile+".mp4")
    t=time.time()
    white_clip = clip1.fl_image(custom_process_image)
    white_clip.write_videofile(white_output, audio=False)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to process test video ...')


#train_model()
#test_on_image(filename="test_images/test1.jpg")
test_on_video("project_video")
#test_on_video("project_video")


