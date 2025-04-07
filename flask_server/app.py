from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from collections import Counter

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the CNN model (same as in your code)
class SkinCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 13)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the class names
class_names = [
    'Basal Cell Carcinoma',
    'Squamous Cell Carcinoma',
    'Melanoma',
    'Actinic Keratosis',
    'Pigmented Benign Keratosis',
    'Seborrheic Keratosis',
    'Vascular Lesion',
    'Melanocytic Nevus',
    'Dermatofibroma',
    'Elastosis Perforans Serpiginosa',
    'Lentigo Maligna',
    'Nevus Sebaceus',
    'Blue Naevus'
]

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkinCNN().to(device)

# In a production environment, you would load your trained model weights here
# model.load_state_dict(torch.load("best_model.pth", map_location=device))
# model.eval()

# For demo purposes, we'll just use the model without weights
# In a real application, you would uncomment the lines above
model.eval()

# Define image transformations (same as in your test_transforms)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get images from the request
        image_files = []
        for i in range(3):  # Check for up to 3 images
            if f'image{i}' in request.files:
                image_files.append(request.files[f'image{i}'])
        
        if not image_files:
            return jsonify({'error': 'No images provided'}), 400
        
        # Process each image and get predictions
        predictions = []
        for img_file in image_files:
            # Read and process the image
            img_bytes = img_file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_tensor = test_transforms(img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                predictions.append(class_names[pred_idx])
        
        # Get the most common prediction (mode)
        if len(predictions) == 1:
            final_prediction = predictions[0]
        else:
            # Find the most common prediction
            prediction_counts = Counter(predictions)
            final_prediction = prediction_counts.most_common(1)[0][0]
        
        return jsonify({
            'prediction': final_prediction,
            'all_predictions': predictions
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

