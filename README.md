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
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the sections `1.1 Initialize images and labels` and `1.2 Extract image features` of the IPython notebook.  

I started by loading all the `vehicle` and `non-vehicle` images and splitting them in a training and a test set. I split the data for vehicle images manually based on the filename (alphabetically) to prevent overfitting. Typically there are several almost identical images in an alphabetical row in the folders. A randomized split therefore would result in similar images in the training and test set, which would result in overfitting.  Here is an example of one image from the `vehicle` and one image from the `non-vehicle` class:

![alt text][image01]

After loading the images I extracted their features. I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feeling for what the `skimage.hog()` output looks like.

Eventually, I found the `YCrCb` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`to work best on my pipeline. Here is a visualized example of hog features based on my customization:

![alt text][image02]

Beyond the hog features, I added color features as it significantly improved the performance of my classifier (section `1.2 Extract image features` first code box).

#### 2. Explain how you settled on your final choice of HOG parameters.

I initially tried some combinations of parameters. When I had a working model with sufficient results I continued to implement my pipeline. Afterwards I returned and fine-tuned the parameters coming up with the parameters described in the previous section. For efficiency purposes I furthermore scaled down the classifier images to (24,24) (section `1.2 Extract image features` first code box).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before training my classifier I applied `StandardScaler()` to standardize my combined hog and color features in section `1.3 Scale training and testing data`.
In section `1.4 Fit and test classifier (SVM)` I trained a linear SVM by applying `LinearSVC()` on the training data. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I realized that cars in certain image areas and therefore distance have very specific sizes. In section `2.1 Define windows` I created `get_window_positions()` and `generate_windows()` which enable a flexibel choice of sizes and densities of windows in different regions of the input image. In the third code cell of the section I defined all the windows I used in my pipeline, which I identified in an experimental approach. Here are all my final 7 search regions on an example image:

![alt text][image03]

Combining all search image regions results in the following windows in the example image:

![alt text][image04]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I optimized the performance of the classifier as described in the previous answers.

Here are the identified windows in the example images as extracted in section `2.2 Get features and predictions for windows`:

![alt text][image05]

In section `2.3 Get image heatmap and return window` I created the functions `get_activated_windows()`, `add_heat()` and `apply_threshold()` in order to draw heatmaps. The heatmaps get activated by incrementing the pixels, which are in an activated window, by 1. The resulting heatmats for the example images look like this:

![alt text][image06]

In section `2.4 Create labeled windows` I enhanced my pipeline by making it possible to plot boxes around the cars. Based on the heatmap images one can use the `get_label()` function, which uses `scipy.ndimage.measurements.label()`, to create a labeled version of the activated fields. Visualized on our test images it looks like this:

![alt text][image07]

Eventually I defined the function `draw_labeled_windows()` in order to draw boxes around the identified areas based on the labels. On our example images the output of `draw_labeled_windows()` looks like this:

![alt text][image08]




---

### Video Implementation

#### 1. Provide a link to your final video output.  

Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In section `3.1 Define class to store previous image values` I defined a class to store the window positions of positive detections in the last frames of the video.


In `3.2 Define function to apply processing pipeline on image` the whole pipeline described in the previous section of this report is implemented. In addition it stores the 10 previously identified windows in the `Windows()` class instance `window_tracker`. For the generation of the heatmap all windows identified in the last 10 images are used to increment the heatmap. After a threshold of 15 is applied on the heatmap, the labels and boxes are generated.



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Strongest issues I faced:
* Identification of search windows
  * Initially I implemented a function which generated different image sizes based on the y-level of the image
  * However, I realized that cars on the left and the right require very different window sizes (wider) than the ones in the middle
  * As a consequence, I implemented my current approach.
* Classifier overfitting
  * Initially I used `sklearn.model_selection.train_test_split()`. 
  * However, I realized that randomization does not work well for the vehicle images, as there are several similar images right after each other
  * Therefore I splitted the vehicle images based on the position in the folder
* Rescaling of images correctly between cv2 and matplotlib-functions
  * Eventually I immediatly rescaled cv2 functions to values between 0 and 1 as matplotlib does for png-images

When is my pipeline likely to fail (and what one could do about it):
* In different light/wheater conditions 
  * Detect light/weather condition
  * Provide customized classifiers for each light/wheather
* In non-flat terrains
  * Implement terrain detection
  * Adjust window frames for search dynamically based on current terrain
* In case of strong traffic on a counter lane right next to the car 
  * Combine vehicle detection with lane detection
  * Only detect cars on lanes, where they are suppose to be
