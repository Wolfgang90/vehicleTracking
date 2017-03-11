**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image01]: ./output_images/01_classifier_images.png
[image02]: ./output_images/02_hog_features_YCrCb.png
[image03]: ./output_images/03_window_areas.png
[image04]: ./output_images/04_image_windows
[image05]: ./output_images/05_predictions.png
[image06]: ./output_images/06_heatmaps.png
[image07]: ./output_images/07_labels.png
[image08]: ./output_images/08_boxes.png

[video01]: ./project_video_annotated.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the sections `1.1 Initialize images and labels` and `1.2 Extract image features` of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images and splitting them in a training and a test set. I split the data for vehicle images manually by spliting it according to the filename to prevent overfitting.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image01]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Eventually, I found the `YCrCb` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`to work best on my pipeline. Here is a visualized example of my hog features:


![alt text][image02]

I combined the hog features with color features as it significantly improved the performance of my classifier (section `1.2 Extract image features` first code box).

####2. Explain how you settled on your final choice of HOG parameters.

I initially tried some combinations of parameters. When I had a working model with sufficient results I continued to implement my pipeline. Afterwards I returned and fine-tuned the parameters coming up with the parameters described in the previous section. For efficiency purposes I furthermore scaled down the classifier images to (24,24) (section `1.2 Extract image features` first code box).

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before training my classifier I applied `StandardScaler()` to standardize my combined hog and color features in section `1.3 Scale training and testing data`.
In section `1.4 Fit and test classifier (SVM)` I trained a linear SVM by applying `LinearSVC()` on the training data. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I realized that cars in certain image areas and therefore distance have very specific sizes. In section `2.1 Define windows` I created `get_window_positions()` and `generate_windows()` which enable a flexibel choice of sizes and densities of windows in different regions of the input image. In the third codesell of the section I defined all the windows I used in my pipeline, which I identified via an experimental approach. Here are all my 7 search regions on one example image:

![alt text][image03]

Combining all search image regions results in the following windows in the example image:

![alt text][image04]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I optimized the performance of the classifier as described in the previous answers.  Here are some example images:

![alt text][image05]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

