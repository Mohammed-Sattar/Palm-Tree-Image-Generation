# Tree Age Prediction and Generation

## Project Overview
This project focuses on predicting the age of trees and generating images of trees based on age-related characteristics using machine learning techniques. It utilizes Generative Adversarial Networks (GANs) and image segmentation models to achieve its goals.


## File Descriptions

### tree_img_generator.py
This file implements a Generative Adversarial Network (GAN) for generating images of trees based on specific noise and age inputs.

**Key Components:**
- **Generator Class**: Takes noise and age as inputs and generates images using a series of linear layers and Leaky ReLU activations.
- **Discriminator Class**: Evaluates the generated images to determine if they are real or fake, also considering the age input.

### cgan_tree_generator.py
This file implements a Conditional Generative Adversarial Network (CGAN) for generating tree images based on age and noise inputs.

**Key Components:**
- **Generator Class**: Similar to the one in tree_img_generator.py, it generates images based on noise and age inputs.
- **Discriminator Class**: Evaluates whether the generated images are real or fake, also considering the age input.

### tree_bg_removal.py
This file is responsible for removing the background from tree images using a segmentation model.

**Key Components:**
- **Model Initialization**: Loads the DeepLabV3 segmentation model pre-trained on a dataset.
- **Preprocessing Function**: Prepares images for segmentation by resizing, normalizing, and converting them to tensors.
- **Post-processing Function**: Processes the output from the segmentation model to create a mask and visualize the segmentation results.

### `tree_age_prediction.ipynb`
A Jupyter Notebook that contains the machine learning model that attempts to predict the age of trees. It uses a Convolutional Neural Network (CNN) architecture to learn patterns in the input images.

### `requirements.txt`
A file that lists the dependencies required for the project. Ensure to install these packages before running the code.

## Installation
To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.