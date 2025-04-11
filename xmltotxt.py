import xml.dom.minidom
from PIL import Image
import os
sort = {
    "good": 1,
    "broke": 2,
    "lose": 3,
    "uncovered": 4,
    "circle": 5,
}

imgfile = "E:\\fwwb\\第三版数据集\\原数据\\uncovered_val"   # 图片
goalDir = "E:\\fwwb\\第三版数据集\\原数据\\uncovered_val_txt"   # txt
originDir = "E:\\fwwb\\第三版数据集\\原数据\\uncovered_val_xml"    #xml
for ori, dirs, files in os.walk(originDir):
    for file in files:
        of = os.path.join(ori, file)
        dom = xml.dom.minidom.parse(of)
        root = dom.documentElement
        imgWid = float(root.getElementsByTagName("width")[0].firstChild.data)
        imgHei = float(root.getElementsByTagName("height")[0].firstChild.data)
        filename, _ = os.path.splitext(file)
        goalFilePath = os.path.join(goalDir, filename+".txt")

        if imgWid == 0 or imgHei == 0:
            relPath = root.getElementsByTagName("path")[0].firstChild.data
            imgPath = os.path.abspath(os.path.join(os.path.dirname(of), relPath))
            img = Image.open(imgPath)
            imgWid, imgHei = img.size
        with open(goalFilePath, 'a') as gf:
            for id in range(len(root.getElementsByTagName("name"))):
                type = root.getElementsByTagName("name")[id].firstChild.data
                xmin = float(root.getElementsByTagName("xmin")[id].firstChild.data)
                ymin = float(root.getElementsByTagName("ymin")[id].firstChild.data)
                xmax = float(root.getElementsByTagName("xmax")[id].firstChild.data)
                ymax = float(root.getElementsByTagName("ymax")[id].firstChild.data)

                typeId = sort.get(type)
                xcen = (xmax+xmin)/2/imgWid
                ycen = (ymax+ymin)/2/imgHei
                wid = (xmax-xmin)/imgWid
                hei = (ymax - ymin)/imgHei
                gf.write(str(typeId)+" "+str(xcen)+" "+str(ycen)+" "+str(wid)+' '+str(hei)+'\n')
