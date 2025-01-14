import shutil
import os

# 原始图片路径
original_image_path ="../data/images/test_img/IMG_20241227_153918.jpg    "


# 目标文件夹路径
destination_folder = '../data/images/test_img2'

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 生成 100 张图片
for i in range(1, 101):
    new_image_name = f'image_{i:03d}.jpg'  # 生成 image_001.jpg, image_002.jpg, ..., image_100.jpg
    new_image_path = os.path.join(destination_folder, new_image_name)
    shutil.copyfile(original_image_path, new_image_path)

print("图片复制和重命名完成。")
