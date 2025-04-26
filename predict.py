import torch
from torchvision import models, transforms
from PIL import Image
import sys

def load_model(path="people_counter_model.pth"):
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image).squeeze().item()
    return round(output)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    model = load_model()
    result = predict(sys.argv[1], model)
    print(f"Предсказано количество людей: {result}")
