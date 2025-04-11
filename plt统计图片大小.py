
import os
import cv2
import matplotlib.pyplot as plt

# 文件夹路径
# 官方数据集 自己数据集
folder_path = 'E:/pic'  # 修改为你的文件夹路径

# 初始化空的 x 和 y 列表
width_list = []
height_list = []

# 循环读取文件夹下所有图片的长和宽
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):  # 确保只读取图片文件
        image_path = os.path.join(folder_path, filename)

        # Attempt to read the image
        img = cv2.imread(image_path)

        # Check if the image was read successfully
        if img is not None:
            height, width, _ = img.shape
            width_list.append(width)
            height_list.append(height)
        else:
            print(f"Unable to read image: {image_path}")

# 绘制散点图
plt.scatter(width_list, height_list, color='red', marker='o', label='size',s=0.5)
plt.xlabel('length', fontdict={'size': 16})
plt.ylabel('width', fontdict={'size': 16})

# 显示图例
plt.legend()

# 显示图形
plt.show()
