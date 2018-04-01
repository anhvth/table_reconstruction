import xml.etree.ElementTree as ET
from glob import glob
import collections
import numpy as np
import cv2

Bndbox = collections.namedtuple("Bndbox", "xmin, xmax, ymin, ymax")


def get_box(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    list_bndbox = []
    for elem in root.iter():
        if 'bndbox' in str(elem):
            xmin, xmax, ymin, ymax = [None] * 4
            for i in elem.iter():
                if 'xmin' in str(i):
                    xmin = int(i.text)

                if 'xmax' in str(i):
                    xmax = int(i.text)
                if 'ymin' in str(i):
                    ymin = int(i.text)
                if 'ymax' in str(i):
                    ymax = int(i.text)

            list_bndbox.append(
                Bndbox(xmin=xmin,
                       xmax=xmax,
                       ymin=ymin,
                       ymax=ymax,
                       )
            )
    return list_bndbox


def draw_cell(image, bndboxs, mode='rectangle'):
    h, w = image.shape[:2]
    image = image.copy() if mode == 'rectangle' else np.zeros(shape=[h, w, 3])
    for bndbox in bndboxs:
        # print(bndbox)
        if mode == 'rectangle':
            cv2.rectangle(image, (bndbox.xmin, bndbox.ymin),
                          (bndbox.xmax, bndbox.ymax), (0, 0, 255))
        elif mode == 'binary':
            ymin, ymax, xmin, xmax = bndbox.ymin, bndbox.ymax, bndbox.xmin, bndbox.xmax
            cx = (xmin+xmax)//2
            cy = (ymin+ymax)//2
            image[ymin:ymax, xmin:xmax, 2] = 255
            # cv2.line(image, (xmin, cy), (xmax, cy), (0, 255, 0), 4)

            image[cy-2:cy+2, xmin:xmax] = (0, 255, 0)
            cv2.rectangle(image, (bndbox.xmin+1, bndbox.ymin+1),
                          (bndbox.xmax-1, bndbox.ymax-1), (0, 255, 255), 2)
    return image.astype(np.uint8)


def cells_to_xml(cells):
    def object_xml(bndbox):
        str_out = '''
        	<object>
		<name>textLine</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
        '''.format(bndbox.xmin, bndbox.ymin, bndbox.xmax, bndbox.ymax)
        return str_out

    list_object_xml = ''.join([object_xml(obj) for obj in cells])

    output_string = '''<annotation>
	<folder>data_cell</folder>
	<filename>0.png</filename>
	<path>/Volumes/Data/Work/Thesis/data/data_cell/0.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1126</width>
		<height>262</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
{}
</annotation>
                    '''.format(list_object_xml)

    return output_string
