In the fall of 2024, I and my teammates did a deep-learning image classification project for STA221. 
The course teaches students about the maths of ML models and students are required to implement the methods taught or beyond in class and develop their engineering skills. 
The problem we were interested in was to assess the statistical reliability of different ML models based on a common rubric. 

The dataset used was HAM 10000, a skin cancer dermoscopy image dataset containing 10015 images. 
Thanks to the advice from Prof. Ricardo Masini and the enlightenment from the seminar talk of Michael Berger, the Head of AI insurance at Munich Re, 
we applied a distribution-free statistical inference tool called Conformal Prediction on the three ML models (logit regression, Inception V3, FixCaps) deployed for the dataset.

The project conclusion is, under the same confidence level, larger deep-learning models, of which the Conformal Prediction procedures produce small prediction sets on average,
outperform the simple multivariate classifier: the logit regression model, which is just a single layer of Neural Network. 
The conclusion, however trivial, is not counter-intuitive and once again proves the power of deep learning models. 

What we learned from the experience is that we got a better understanding of the advantages and pitfalls of Conformal Prediction.
What’s more, we acquired the engineering skills of constructing medium-scale ML model (1-10 M parameters) pipelines 
and using the new GPU cluster “MSBC” in the UC Davis Statistics department. 
