from itertools import product
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

# ========================
# 数据准备与环境设定
# ========================

data_dir = r"101_ObjectCategories"  # 修改为你本地数据集路径
log_dir = r"runs"
seed = 42
torch.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

# 每类取30张图像用于训练，其余用于验证
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


# ========================
# 训练函数
# ========================
def run_experiment(learning_rate_base, learning_rate_fc, num_epochs, run_name):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 101)
    model = model.to(device)

    fc_params = list(model.fc.parameters())
    base_params = [p for name, p in model.named_parameters() if "fc" not in name]
    optimizer = optim.SGD([
        {"params": base_params, "lr": learning_rate_base},
        {"params": fc_params, "lr": learning_rate_fc}
    ], momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=f'{log_dir}/{run_name}')
    best_val_acc = 0.0
    best_model_path = f'{log_dir}/{run_name}/best_model.pth'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

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

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(f"[{run_name}] Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val_acc={val_acc:.4f}")

    # 超参数记录
    writer.add_hparams(
        {
            'lr_base': learning_rate_base,
            'lr_fc': learning_rate_fc,
            'epochs': num_epochs
        },
        {
            'hparam/val_acc': best_val_acc
        }
    )
    writer.close()


# ========================
# 网格搜索配置与运行
# ========================
lr_base_space = [1e-4, 5e-5]
lr_fc_space = [1e-3, 5e-3, 1e-2]
epoch_space = [10, 15]

all_combinations = list(product(lr_base_space, lr_fc_space, epoch_space))
configs = []
for i, (lr_base, lr_fc, epochs) in enumerate(all_combinations, start=1):
    configs.append({
        "learning_rate_base": lr_base,
        "learning_rate_fc": lr_fc,
        "num_epochs": epochs,
        "run_name": f"exp{i}"
    })

for cfg in configs:
    run_experiment(**cfg)
