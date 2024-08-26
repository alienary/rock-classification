import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = models.resnext101_32x8d(pretrained=False)
model.fc = nn.Linear(in_features=2048, out_features=7)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['玄武岩', '煤', '花岗岩', '石灰石', '大理石', '石英岩', '砂岩']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image = Image.open(file_path)
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image)
                _, predicted = outputs.max(1)
                predicted_class = classes[predicted.item()]
            return jsonify({'class': predicted_class})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
