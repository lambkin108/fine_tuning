import os
import random
from PIL import Image

# 数据根目录
root_dir = "101_ObjectCategories"

# 排除背景类
class_dirs = [d for d in os.listdir(root_dir)
              if os.path.isdir(os.path.join(root_dir, d)) and d != "BACKGROUND_Google"]

# 随机从每类中选一张图片
images = []
labels = []

for class_name in sorted(class_dirs):
    class_path = os.path.join(root_dir, class_name)
    img_list = os.listdir(class_path)
    if img_list:
        img_name = random.choice(img_list)
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            labels.append(class_name)
        except:
            continue  # 跳过损坏图像

# 自定义每行多少列
cols = 10
rows = (len(images) + cols - 1) // cols

# 计算每列最大宽度，每行最大高度
col_widths = [0] * cols
row_heights = [0] * rows

for idx, img in enumerate(images):
    row, col = divmod(idx, cols)
    w, h = img.size
    col_widths[col] = max(col_widths[col], w)
    row_heights[row] = max(row_heights[row], h)

# 计算总画布大小
canvas_width = sum(col_widths)
canvas_height = sum(row_heights)

# 创建白色画布
canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

# 逐图粘贴
x_offsets = [0] * cols
y_offsets = [0] * rows
for c in range(1, cols):
    x_offsets[c] = x_offsets[c-1] + col_widths[c-1]
for r in range(1, rows):
    y_offsets[r] = y_offsets[r-1] + row_heights[r-1]

for idx, img in enumerate(images):
    row, col = divmod(idx, cols)
    x = x_offsets[col]
    y = y_offsets[row]
    canvas.paste(img, (x, y))

# 保存或显示结果
canvas.save("caltech101_collage_rawsize.jpg")
canvas.show()
