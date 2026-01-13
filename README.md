This repository contains two neural network projects developed for our Machine Learning final project and our second Capstone project.

OrganNet was trained using the OrganMNIST3D dataset. The final model achieved 93.9% test accuracy with a test loss of 0.18. This project was submitted as our Machine Learning course final.

FaceNet was trained on the WIDER Face dataset. After training, the model is saved and loaded for use in a live camera feed via OpenCV.

Due to a short development timeline (approximately one week) and additional constraints, the scope was reduced to training on images containing a single face. On the validation set, the model detected faces in approximately 5% of images. However, in cases where a face was detected, it achieved an average Intersection over Union (IoU) of ~60%, indicating stronger localization performance than detection consistency.

The FaceNet project will be updated in the future to improve detection performance and expand its scope.
