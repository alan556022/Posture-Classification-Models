# Image Classification Models to Improve Posture Classifcation
This repository consists of data visualizations and documentation of our model implementations for predicting postures of subjects in images. We also used have code to process videos to load into models for posture prediction.

## [MobileNet](https://github.com/alan556022/Posture-Classification-Models/blob/master/MobileNetModel.h5)
Since the blank MobileNet model (random initialization) will cause overfitting, we tried MobileNet pre-trained on imagenet, froze its convolution layers, and retrained its fully connected layers on our data. Our data set is all the three tranches excluding the null and unknown values, with about 36k images. We trained on 30 thousand images with a 0.5 validation split and tested the model on the remaining 6 thousand. The model was trained on Google Colab. The results are as follows:

<img src="https://github.com/alan556022/Posture-Classification-Models/blob/master/visuals/mobilenet1_1.png" alt="mobilenet1" width="450"/>
<img src="https://github.com/alan556022/Posture-Classification-Models/blob/master/visuals/mobilenet2.png" alt="mobilenet2" width="450"/>

## Inception-ResNet v2
Similar to MobileNet, we used the same training, validation, and test set. And we froze the convolutional layers pre-trained on imagenet and retrained the fully connected layers. The model was trained on Google Colab. The results are as follows:

<img src="https://github.com/alan556022/Posture-Classification-Models/blob/master/visuals/inceptionresnetv2.png" alt="inceptionresnetv2" width="450"/>

## MobileNet + Edge Detection
We also implemented some preprocessing techniques. The edge detection code was from group 2. We combine the original image and the edge-detected image together into the training set. The following models were trained on the CARC system from USC. We tried MobileNet pre-trained on imagenet, first training on the original images. Then we trained the model on edge-detected images combined with original images to see if there is any improvement.

### Original Images
<img src="https://github.com/alan556022/Posture-Classification-Models/blob/master/visuals/mobilenet_edgedet.png" alt="mobilenet_edgedet1" width="450"/>

### [Edge-detected + Original Images](https://github.com/alan556022/Posture-Classification-Models/blob/master/MobileNet%2BEdgeDetection_Frozen.h5)
<img src="https://github.com/alan556022/Posture-Classification-Models/blob/master/visuals/mobilenet_edgedet2.png" alt="mobilenet_edgedet2" width="450"/>

As a result, we don’t see considerable improvement in accuracy. It may be due to the following reasons: the convolutional layers are frozen or edge detection is not effective. In order to check the reason, we unfroze the convolutional layers in MobileNet and trained the model.

### MobileNet Unfrozen Without Edge Detection
<img src="https://github.com/alan556022/Posture-Classification-Models/blob/master/visuals/mobilenet_edgedet3.png" alt="mobilenet_edgedet3" width="450"/>

### [MobileNet Unfrozen With Edge Detection](https://github.com/alan556022/Posture-Classification-Models/blob/master/MobileNet-Edged-Unfrozen.h5)
<img src="https://github.com/alan556022/Posture-Classification-Models/blob/master/visuals/mobilenet_edgedet4.png" alt="mobilenet_edgedet4" width="450"/>

Surprisingly, the accuracy increases as we unfroze the layers. The validation accuracy is now about 90%.

## [Processing Videos for Posture Prediction](https://github.com/alan556022/Posture-Classification-Models/blob/master/Video_Cutting_andPrediction.ipynb)

In this file, we process a video by cutting the video into individual frames as a list of images in order to make posture predictions. We can load any of the above models to make predictions.

## Summary Table
<img src="https://github.com/alan556022/Posture-Classification-Models/blob/master/visuals/summary.png" alt="summary" width="800"/>

## [Other Visualizations](https://github.com/alan556022/Posture-Classification-Models/blob/master/visualizations_doc.ipynb)
We created some additional visualizations in the visualizations_doc.ipynb file linked. Our goal was to see if the distribution of images labeled occluded or not is significant, and if there are noticeble patterns dimensions of the images in relation to the image's primary_posture label.

Findings: The data is unbalanced in that the majority of the sample images are labeled with 'Standing', while the 'Lying' label is the smallest in terms of the number of pictures labeled with. Further, the majority of images labeled with 'Standing' have the highest height-to-width ratios, whereas those labeled with 'Lying' have the lowest. Additionally, although images that are not occluded seem to have higher height-to-width ratios than the occluded ones for 'Standing' and 'Sitting' labels, such a trend not longer holds for images labeled with 'Lying' probably because there are much fewer pictures labeled with 'Lying' than those labeled with 'Standing' or 'Sitting'. Furthermore, there exists a potential relationship between the number of people each image contains (from 0 to 3) and its height-to-width ratio: the more the people 1 picture contains, the lower its height-to-width ratio would be across all 3 postures labels ('Standing', 'Sitting', and 'Lying'). As a result, it might be worth splitting the data according the number of people (in each image) before preprocessing and then running them in a model.

## Limitations and Improvements
Ratio of the pictures was affected by posture and the amount of people. The pictures with standing had the highest ratio, followed by sitting and lying. The fewer the people, the higher the ratio. Also, the non-occluded pictures had higher ratios as well.

One of the improvements that our group found was dropping out the images with extreme ratios. This is because the outliers in picture ratios can dilute the actual results. Also, we found that object detitions could be a better improvement in results. Focusing on the center of the object, instead of the edges could lead to higher accuracy then the edge-detection system. Furthermore, taking a region of the graph instead of resizing the whole image could lead to a better accuracy. Finally, we decided we decided that we would split the loudness and/or number of people in the image for more complete data. 

