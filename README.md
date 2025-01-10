# A Brief Introduction to the project:
In the fall of 2024, We (ChiChun Chen, Savali Sandip Deshmukh, Shivani Sanjay Suryawanshi, Sriharshini D, and Yehong Qiu) did a deep-learning image classification project for STA221. The course teaches students about the maths of Machine Learning models and students are required to implement the methods taught or beyond in class and develop their engineering skills in the process. 

The problem we were interested in was to assess the statistical reliability of different ML models based on a common rubric. 

The dataset used was HAM 10000, a skin cancer dermoscopy image dataset containing 10015 images. The goal for machine learning training is to classify the skin tumor within each image to 1 out of the 7 classes: Actinic keratoses and intraepithelial carcinoma (akiec), Basal cell carcinoma (bcc), Benign keratosis-like lesions (bkl), Dermatofibroma (df), Melanoma (mel), Melanocytic nevi (nv), and Vascular lesions (vasc). 
![dataset glimpse](figs/dataset.png "A glimpse of the HAM10000 Dataset")
The dataset is an unbalanced dataset in which the number of images under each class varies greatly. 
![dataset detail](figs/dataset_ub.png "Unbalanced data in the HAM10000 Dataset")
After training three Machine Learning models (logit regression, Inception V3, FixCaps) on the dataset, we used an additional dataset of 1151 images unseen during training to validate the models' capability. 
 ![dataset detail](figs/ova.png "Overall accuracy")
 <p align="center">
  <img src="figs/logit.png" alt="logit regression" width="200"/>
  <img src="figs/iv3.png" alt="inception v3" width="200"/>
  <img src="figs/fixcaps.png" alt="fixcaps" width="200"/>
</p>

To assess the statistical reliability of the models, we applied a distribution-free statistical inference tool called Conformal Prediction on the three ML models.
<p align="center">
  <img src="figs/sps_logit.png" alt="logit regression" width="200"/>
  <img src="figs/sps_iv3.png" alt="inception v3" width="200"/>
  <img src="figs/sps_fixcaps.png" alt="fixcaps" width="200"/>
</p>

<p align="center">
  <img src="figs/ec.png" alt="empirical coverage" width="300"/>
  <img src="figs/sps.png" alt="prediction set size" width="300"/>
</p>
<p align="center">
  <img src="figs/csc.png" alt="class-stratified coverage" width="300"/>
  <img src="figs/ssc.png" alt="size-stratified coverage" width="300"/>
</p>

The project conclusion is, under the same confidence level, larger deep-learning models, of which the Conformal Prediction procedures produce small prediction sets on average, outperform the simple multivariate classifier: the logit regression model, which is just a single layer of Neural Network. 
The conclusion, however trivial, is not counter-intuitive and once again proves the power of deep learning models. 

What we learned from the experience is that we got a better understanding of the advantages and pitfalls of Conformal Prediction.
What’s more, we acquired the engineering skills of constructing medium-scale ML model (1-10 M parameters) pipelines 
and using the new GPU cluster “MSBC” in the UC Davis Statistics department. 

We thank Prof. Ricardo Masini for his enlightful lectures and advice on this project, and Michael Berger, the Head of AI insurance at Munich Re, for the seminar talk in which he shared his industry-wise pioneering team  use Conformal Prediction to design insurance plans for AI companies selling black-box-like AI tools.

Becides, we referenced:
1. https://github.com/Woodman718/FixCaps
2. https://github.com/aangelopoulos/conformal-prediction
to produce our codes. A detailed list of references is in the report.
