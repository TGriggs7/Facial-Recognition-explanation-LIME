# Facial-Recognition-explanation-LIME
Using the pre-trained FaceNet model for facial recognition, I apply lime to the model to investigate which facial features most significantly contribute to the models decision. This creates an image mask colored green or red depending on how significant the positive/negative effect of a given part of the image was on the final model decision.

The facial recognition explanation tool can be used in a jupyter notebook or simply running the python file 'faceRecogExplainer.py'.

It is recommended to use the jupyter notebook, as it takes a while for the model weights to be loaded and the jupyter notebook makes it simpler to play around with the explaination tool multiple times while only loading the model once.
