from datasets.vrd import VRDDataset
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
import datasets.transforms as T
from PIL import Image
import torch

def make_voc_transforms(image_set):
	
	normalize = T.Compose([
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

	if image_set == 'train':
		return T.Compose([
			T.RandomHorizontalFlip(),
			T.RandomSelect(
				T.RandomResize(scales, max_size=1333),
				T.Compose([
					T.RandomResize([400, 500, 600]),
					T.RandomSizeCrop(384, 600),
					T.RandomResize(scales, max_size=1333),
				])
			),
			normalize,
		])

	if image_set == 'test':
		return T.Compose([
			T.RandomResize([800], max_size=1333),
			normalize,
		])

	raise ValueError(f'unknown {image_set}')


def _findNode(parent, name, debug_name=None, parse=None):
	if debug_name is None:
		debug_name = name

	result = parent.find(name)
	if result is None:
		raise ValueError('missing element \'{}\''.format(debug_name))
	if parse is not None:
		try:
			return parse(result.text)
		except ValueError as e:
			raise_from(ValueError(
				'illegal value for \'{}\': {}'.format(debug_name, e)), None)
	return result


class VOCDataset:

	def __init__(self, root, image_set, transform=None):
		"""Dataset for VOC data.
		Args:
				root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
						Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
		"""
		self.root = pathlib.Path(root)
		self.transform = transform
		if image_set == 'test':
			image_sets_file = self.root / "ImageSets/Main/test.txt"
		else:
			image_sets_file = self.root / "ImageSets/Main/trainval.txt"
		self.ids = VOCDataset._read_image_ids(image_sets_file)
	
		logging.info("No labels file, using default VOC classes.")
		
		self.class_names = ('bicycle', 'truck', 'tt', 'bus', 'car', 'motorbike', 'autorickshaw')

		
		self.class_dict = {class_name: i for i,
						   class_name in enumerate(self.class_names)}
		
		# remove ids
		self.remove_ids()
	


	# def remove_ids(self):
	# 	image_names_copy=self.ids.copy()
	# 	for image_name in self.ids:
	# 		filename = os.path.join(image_name + '.xml')
	# 		tree = ET.parse(os.path.join(self.root, 'Annotations', filename))
	# 		annotations = self.__parse_annotations(tree.getroot())
	# 		if (annotations['labels'].size == 0):
	# 				image_names_copy.remove(image_name)
	# 	self.ids = image_names_copy

	def remove_ids(self):
		mask = []
		image_names_copy=self.ids.copy()
		for image_name in self.ids:
			filename = os.path.join(image_name + '.xml')
			tree = ET.parse(os.path.join(self.root, 'Annotations', filename))
			annotations = self.__parse_annotations(tree.getroot())
			mask.append(annotations['labels'].size != 0)
		
		mask = np.array(mask)
		self.ids = np.array(self.ids)
		self.ids = self.ids[mask]


	def __parse_annotation(self, element):
		""" Parse an annotation given an XML element.
		"""
		# truncated = _findNode(element, 'truncated', parse=int)
		# difficult = _findNode(element, 'difficult', parse=int)

		class_name = _findNode(element, 'name').text
		if class_name not in self.class_dict:
			return None, None

		box = []
		label = self.class_dict[class_name]

		bndbox = _findNode(element, 'bndbox')
		box.append(_findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1)
		box.append(_findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1)
		box.append(_findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1)
		box.append(_findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1)

		return box, label

	def __parse_annotations(self, xml_root):
		""" Parse all annotations under the xml_root.
		"""
		annotations = {'labels': [], 'bboxes': []}

		for i, element in enumerate(xml_root.iter('object')):
			try:
				box, label = self.__parse_annotation(
					element)
				if label is None:
					continue
			except ValueError as e:
				raise_from(ValueError(
					'could not parse object #{}: {}'.format(i, e)), None)

			annotations['bboxes'].append(box)
			annotations['labels'].append(label)

		annotations['bboxes'] = np.array(annotations['bboxes'])
		annotations['labels'] = np.array(annotations['labels'])

		return annotations

	def __getitem__(self, index):
		image_id = self.ids[index]
		boxes, labels = self._get_annotation(image_id)
		# if not self.keep_difficult:
		# 	boxes = boxes[is_difficult == 0]
		# 	labels = labels[is_difficult == 0]
		image = self._read_image(image_id)
		target = {"boxes": boxes, "labels": labels,}
		img, target= self.transform(image, target)
		return img, target

	def get_image(self, index):
		image_id = self.ids[index]
		image = self._read_image(image_id)
		if self.transform:
			image, _ = self.transform(image)
		return image

	def get_annotation(self, index):
		image_id = self.ids[index]
		return image_id, self._get_annotation(image_id)

	def __len__(self):
		return len(self.ids)

	@staticmethod
	def _read_image_ids(image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip())
		return ids

	def _get_annotation(self, image_id):
		annotation_file = self.root / f"Annotations/{image_id}.xml"
		objects = ET.parse(annotation_file).findall("object")
		boxes = []
		labels = []
		is_difficult = []
		for object in objects:
			class_name = object.find('name').text.lower().strip()
			# we're only concerned with clases in our list
			if class_name in self.class_dict:
				bbox = object.find('bndbox')

				# VOC dataset format follows Matlab, in which indexes start from 0
				x1 = float(bbox.find('xmin').text) - 1
				y1 = float(bbox.find('ymin').text) - 1
				x2 = float(bbox.find('xmax').text) - 1
				y2 = float(bbox.find('ymax').text) - 1
				boxes.append([x1, y1, x2, y2])

				labels.append(self.class_dict[class_name])
	
		return (torch.tensor(boxes, dtype=torch.float32),
				torch.tensor(labels, dtype=torch.int64))

	def _read_image(self, image_id):
		image_file = self.root / f"JPEGImages/{image_id}.jpg"
		image = Image.open(image_file).convert('RGB')
		return image

def build(image_set, args):
	# root = Path(args.coco_path)
	# assert root.exists(), f'provided COCO path {root} does not exist'
	# mode = 'instances'
	# PATHS = {
	#     "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
	#     "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
	# }

	# img_folder, ann_file = PATHS[image_set]
	dataset = VOCDataset(args.voc_path , image_set, make_voc_transforms(image_set))
	return dataset
