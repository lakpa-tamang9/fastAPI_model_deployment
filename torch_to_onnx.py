import torch
import torch.onnx
from model import NeuralNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# A model class instance (class not shown)
FILE = "./models/model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(17)

torch.onnx.export(model, dummy_input, "./models/sejam_onnx.onnx")