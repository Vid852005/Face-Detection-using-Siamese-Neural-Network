# Face-Detection-using-Siamese-Neural-Network
guys, faceid is the final code face detection is to code siamesenn.keras which was too heavy to upload
Anchor= using webcam
Positive=Webcam
Negative=Labelled faces in wild
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

in the first sister network, we are comparing the anchor with negative and passing it through an embedding layer to train it that , the output should be 0 meaning not matching and then we go to other sister layer in which we are doing the same for positive, output 1 and then both these layers are passed through subtraction layer.

research paper-efaidnbmnnnibpcajpcglclefindmkaj/https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
