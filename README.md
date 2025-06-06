# Fine_tuning
Fine-tuning ResNet18 pretrained on ImageNet for image classification on the Caltech-101 dataset.

## Structure
```
fine_tuning/
├── baseline.py                # without pretrain
├── train_grid_search.py       # fine_tuning_train + Grid Search
├── test_model.py              # Evaluation Script
├── split_indices.json         # Stored train/val split
├── runs/                      # Tensorboard logs and best_model
│   ├── exp1/
│   └── ...
└── 101_ObjectCategories/      # Caltech-101 dataset
```

## Training
```
python train_grid_search.py
```

This script will:
+ Use 30 samples per class for training
+ Use the remaining samples for validation
+ Automatically save best models and TensorBoard logs under runs/exp*/
+ Save the split indices to split_indices.json to avoid data leakage in testing


## Testing
**Download the pre-trained model weights from the following link:**

链接: https://pan.baidu.com/s/1yFqcpvE0EdwpSeHqQcPAbA 提取码: udss 

```
python test_model.py
```

This script will:
+ Loads the validation set using the same split_indices.json
+ Loads the model from the path you saved
+ Reports final test accuracy without data leakage
  
**Note: Make sure the paths to the following two files are correct.**


Final Test Accuracy: 0.9473

## 
