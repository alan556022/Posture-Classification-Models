# Math499
This repository consists of the documentation of our model implementations and data visualizations.

## MobileNet
Since the blank MobileNet model (random initialization) will cause overfitting, we tried MobileNet pre-trained on imagenet, froze its convolution layers, and retrained its fully connected layers on our data. Our data set is all the three tranches excluding the null and unknown values, with about 36k images. We trained on 30 thousand images with a 0.5 validation split and tested the model on the remaining 6 thousand. The model was trained on Google Colab. The results are as follows:
![https://github.com/alan556022/Math499/edit/master/visuals/mobilenet1.png]
![https://github.com/alan556022/Math499/edit/master/visuals/mobilenet2.png]

## Inception-ResNet v2
Similar to MobileNet, we used the same training, validation, and test set. And we froze the convolutional layers pre-trained on imagenet and retrained the fully connected layers. The model was trained on Google Colab. The results are as follows:
![https://github.com/alan556022/Math499/edit/master/visuals/inceptionresnetv2.png]

## MobileNet + Edge Detection
We also implemented some preprocessing techniques. The edge detection code was from group 2. We combine the original image and the edge-detected image together into the training set. The following models were trained on the CARC system from USC. We tried MobileNet pre-trained on imagenet, first training on the original images. Then we trained the model on edge-detected images combined with original images to see if there is any improvement.
### Original Images
![https://github.com/alan556022/Math499/edit/master/visuals/mobilenet_edgedet1.png]
### Edge-detected + Original Images
![https://github.com/alan556022/Math499/edit/master/visuals/mobilenet_edgedet2.png]

As a result, we don’t see considerable improvement in accuracy. It may be due to the following reasons: the convolutional layers are frozen or edge detection is not effective. In order to check the reason, we unfroze the convolutional layers in MobileNet and trained the model.
### MobileNet Unfrozen Without Edge Detection
![https://github.com/alan556022/Math499/edit/master/visuals/mobilenet_edgedet3.png]
### MobileNet Unfrozen With Edge Detection
![https://github.com/alan556022/Math499/edit/master/visuals/mobilenet_edgedet4.png]

Surprisingly, the accuracy increases as we unfroze the layers. The validation accuracy is now about 90%.

# Summary Table
![https://github.com/alan556022/Math499/edit/master/visuals/summary.png]

# Other Visualizations

# Limitations and Improvements
Ratio of the pictures was affected by posture and the amount of people. The pictures with standing had the highest ratio, followed by sitting and lying. The fewer the people, the higher the ratio Also, the non-occluded pictures had higher ratios as well.

One of the improvements that our group found was dropping out the images with extreme ratios. This is because the outliers in picture ratios can dilute the actual results. Also, we found that object detitions could be a better improvement in results. Focusing on the center of the object, instead of the edges could lead to higher accuracy then the edge-detection system. Furthermore, taking a region of the graph instead of resizing the whole image could lead to a better accuracy. Finally, we decided we decided that we would split the loudness and/or number of people in the image for more complete data. 

