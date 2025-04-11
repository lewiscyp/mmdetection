import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse


def parse_opt():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--annotation_path', type=str, default='/root/LLVIP/Annotations',
                        help='folder containing xml files')
    parser.add_argument('--json_save_path', type=str, default='/root/LLVIP/LLVIP.json', help='json file')
    opt = parser.parse_args()
    return opt


opt = parse_opt()

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

# 预定义类别列表
categories = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
category_set = {name: idx + 1 for idx, name in enumerate(categories)}  # 使用字典保存类别名称和对应的 ID
image_set = set()

image_id = 0
annotation_id = 0

# 添加类别到 coco['categories']
def addCategoryItem():
    for category, category_id in category_set.items():
        category_item = dict()
        category_item['id'] = category_id
        category_item['name'] = category
        coco['categories'].append(category_item)


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_id += 1
    annotation_item['id'] = annotation_id
    annotation_item['category_id'] = category_id  # 使用预定义类别 ID
    coco['annotations'].append(annotation_item)


def parseXmlFiles(xml_path, json_save_path):
    # 确保目标文件夹存在
    if not os.path.exists(os.path.dirname(json_save_path)):
        os.makedirs(os.path.dirname(json_save_path))

    # 添加类别到 coco
    addCategoryItem()

    for f in tqdm(os.listdir(xml_path)):
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text

            # 只有解析完 <size> 标签后才添加图像项
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                else:
                    raise Exception('duplicated image: {}'.format(file_name))

            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        raise Exception(f"Unknown object category: {object_name}")
                    current_category_id = category_set[object_name]  # 使用预定义类别 ID

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    bbox.append(bndbox['xmin'])
                    bbox.append(bndbox['ymin'])
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)

    # 保存 JSON 文件
    with open(json_save_path, 'w') as json_file:
        json.dump(coco, json_file)


if __name__ == '__main__':
    ann_path = opt.annotation_path
    json_save_path = opt.json_save_path
    parseXmlFiles(ann_path, json_save_path)
