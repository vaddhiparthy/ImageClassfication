# ImageClassfication

PyTorch Image Classification with a Pre-Trained ResNet-18 Model
Overview
This code demonstrates how to use PyTorch to classify images using a pre-trained ResNet-18 model. The model is loaded from the PyTorch model zoo and is applied to a sample image to predict its class label.

Dependencies
This code requires the following dependencies:

PyTorch
torchvision
Pillow
These dependencies can be installed via pip:

bash
Copy code
pip install torch torchvision Pillow
Usage
To classify an image, simply run the following command:

bash
Copy code
python classify_image.py --image_path cat.jpg
The --image_path argument specifies the path to the input image.

The script will apply the pre-processing transformation to the input image, pass it through the ResNet-18 model, and output the predicted class label.

Example Output
When run on the cat.jpg image, the script outputs:

python
Copy code
Predicted class: cat
