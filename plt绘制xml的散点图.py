
import xml.etree.ElementTree as ET
import os

import matplotlib.pyplot as plt
folder_path = 'E:/all_xml'  # 替换成你的XML文件所在的文件夹路径
x = []
y = []
def read_bndbox_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall('.//object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        dx = xmax - xmin
        dy = ymax - ymin
        x.append(dx)
        y.append(dy)

def process_all_xml_files(folder_path):
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

    for xml_file in xml_files:
        xml_file_path = os.path.join(folder_path, xml_file)
        read_bndbox_from_xml(xml_file_path)


if __name__ == "__main__":
    process_all_xml_files(folder_path)
    plt.scatter(x, y, color='red', marker='o', label='bndbox size', s=0.5)

    plt.xlabel('bndbox length', fontdict={'size': 16})
    plt.ylabel('bndbox width', fontdict={'size': 16})

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

