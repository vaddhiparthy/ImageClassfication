import torch
import torchvision
from torchvision import transforms

# Define the transformation to be applied to the input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the pre-trained model
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Define the list of class labels
labels = ['cat', 'dog', 'horse', 'bird', 'sheep']

# Load the image
img = Image.open("cat.jpg")

# Apply the transformation and add a batch dimension to the input image
input_tensor = transform(img).unsqueeze(0)

# Pass the input image through the model and get the predicted class label
with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)
    predicted_label = labels[pred[0]]

# Print the predicted class label
print(f"Predicted class: {predicted_label}")
