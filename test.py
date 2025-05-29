import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Subset
import json

data_dir = r"101_ObjectCategories"
log_dir = r"runs"
model_path = "runs/exp4/best_model.pth"  # 替换为你要测试的模型路径
split_path = "split_indices.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
with open(split_path, "r") as f:
    indices = json.load(f)
val_indices = indices["val_indices"]

# 构建验证集
val_set = Subset(full_dataset, val_indices)
val_loader = DataLoader(val_set, batch_size=32)
# 构建与加载模型
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 101)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 测试模型准确率
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nFinal Test Accuracy: {correct / total:.4f}")
