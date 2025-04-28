from flask import Flask, request, jsonify
from flasgger import Swagger
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io
import base64

app = Flask(__name__)
swagger = Swagger(app)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")

# Model setup
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("gender_classifier_pytorch.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/gender', methods=['POST'])
def predict():
    """
    Predict Gender from Base64 Image
    ---
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            image_base64:
              type: string
              description: Base64 encoded image string
              example: "iVBORw0KGgoAAAANSUhEUgAABVYAAAJRC..."
    responses:
      200:
        description: Predicted gender with confidence
        schema:
          type: object
          properties:
            prediction:
              type: string
              example: "Male"
            confidence:
              type: number
              example: 0.9876
    """
    data = request.json
    img_data = base64.b64decode(data['image_base64'])
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = int(output.item() > 0.5)
        confidence = float(output.item())
        label = "Male" if pred else "Female"

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
