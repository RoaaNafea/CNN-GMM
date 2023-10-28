# CNN-GMM
This work introduces a novel methodology that employs a Convolutional Neural Network (CNN) and Gaussian Mixture Model (GMM) to effectively differentiate between fake and real images or videos.
The proposed methodology presents a novel CNN-GMM architecture, in which the fully connected (FC) layer in the CNN is replaced with a customized Gaussian Mixture Model (GMM) fully connected layer.
The GMM Layer utilizes a weighted set of Gaussian Probability Density Functions (PDFs) to represent the distribution of data frequencies in both real and fake images.
This representation indicates there is a shift in the distribution of the manipulated images due to added noise.
The CNN-GMM model demonstrates the ability to accurately identify variations resulting from various types of deepfakes within the probability distribution.
The GMM layer was developed based on Blundell et al. work in "Weight Uncertainty in Neural Networks" paper.

Preprocessing code include preprocessing steps for FaceForensics++ data set.

Pixel_PCA code apply PCA to the extracted images.

CNN-GMM include proposed topology.

GMM Fully Connected shows GMM Layer.
