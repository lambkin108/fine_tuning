import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

data_dir = r"101_ObjectCategories"
log_dir = r"runs"

seed = 42
torch.manual_seed(seed)
random.seed(seed)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

# 每类30张用于训练
class_indices = {i: [] for i in range(len(full_dataset.classes))}
for idx, (_, label) in enumerate(full_dataset):
    class_indices[label].append(idx)

train_indices, val_indices = [], []
for idx_list in class_indices.values():
    random.shuffle(idx_list)
    train_indices += idx_list[:30]
    val_indices += idx_list[30:]

train_set = Subset(full_dataset, train_indices)
val_set = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建模型：不使用预训练，从头训练 ResNet18
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 101)  # 设置输出层为101类
model = model.to(device)

# 设置优化器（统一学习率，因从头训练）
lr=1e-3
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir='runs/exp0')
best_val_acc = 0.0
best_model_path = f'{log_dir}/best_model.pth'
# 训练
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    # TensorBoard 可视化
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    print(f"Epoch [{epoch+1}/{epochs}]  "
          f"Train Loss: {avg_train_loss:.4f}  "
          f"Val Loss: {avg_val_loss:.4f}  "
          f"Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved at epoch {epoch+1} with val_acc={val_acc:.4f}")

writer.add_hparams(
    {
        'lr': lr,
        'epochs': epochs
    },
    {
        'hparam/val_acc': best_val_acc
    }
)

writer.close()

