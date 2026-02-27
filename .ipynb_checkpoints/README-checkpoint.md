# cnn-gradcam-tiny-imagenet
COGS 181 Final Project: Using Grad-CAM to analyse various CNN representations on the Tiny ImageNet dataset.

Files included:
1. dataset.py: loads dataset
2. model.py: builds the model
3. train.py: conducts training loop and optimizer
4. gradcam.py: contains Grad-CAM class
5. cnn_gradcam.ipynb: runs experiments and does visualisations

### Main idea:
- Training a convolutional neural network to classify images from the Tiny ImageNet dataset
- Using Grad-CAM to interpret what the network was actually learning across each layer
- Analysing effects of various parameters (optimizer, normalisation method, deeper networks, etc.) 