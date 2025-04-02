# model/inference.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

from train import SimpleCNN  # or define SimpleCNN here if you prefer

class MNISTModel:
    def __init__(self, model_path="model.pkl"):
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def predict(self, image: Image.Image):
        # image is a PIL Image
        tensor = self.transform(image).unsqueeze(0)  # shape: [1, 1, 28, 28]
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        return predicted_class, confidence
