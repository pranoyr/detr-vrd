import json
import os
import torch
from util.box_ops import boxes_union
from pathlib import Path
from PIL import Image
import datasets.transforms as T
from torch.utils.data import Dataset
from util.box_ops import y1y2x1x2_to_x1y1x2y2


def make_image_list(dataset_path, type):
	imgs_list = []
	with open(os.path.join(dataset_path, 'json_dataset', f'annotations_{type}.json'), 'r') as f:
		annotations = json.load(f)
	sg_images = os.listdir(os.path.join(
		dataset_path, 'sg_dataset', f'sg_{type}_images'))

	annotations_copy = annotations.copy()
	for ann in annotations.items():
		if(not annotations[ann[0]] or ann[0] not in sg_images):
			annotations_copy.pop(ann[0])

	for ann in annotations_copy.items():
		imgs_list.append(ann[0])
	return imgs_list


class VRDDataset(Dataset):
	"""VRD dataset."""

	def __init__(self, dataset_path, image_set):
		self.dataset_path = dataset_path
		self.image_set = image_set
		# read annotations file
		with open(os.path.join(self.dataset_path, 'json_dataset', f'annotations_{self.image_set}.json'), 'r') as f:
			self.annotations = json.load(f)
		with open(os.path.join(self.dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
			self.all_objects = json.load(f)
		with open(os.path.join(self.dataset_path, 'json_dataset', 'predicates.json'), 'r') as f:
			self.predicates = json.load(f)

		self.root = os.path.join(
			self.dataset_path, 'sg_dataset', f'sg_{self.image_set}_images')

		self.classes = self.all_objects.copy()
		self.preds = self.predicates.copy()
		# self.classes.insert(0, '__background__')
		# self.preds.insert(0, 'unknown')

		self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
		self._preds_to_ind = dict(zip(self.preds, range(len(self.preds))))
		self.imgs_list = make_image_list(self.dataset_path, self.image_set)

		# print(self._class_to_ind)
		# print(self._preds_to_ind)

		normalize = T.Compose([
			T.ToTensor(),
			T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

		scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

		if image_set == 'train':
			self.transform =  T.Compose([
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
			self.transform =  T.Compose([
				T.RandomResize([800], max_size=1333),
				normalize,
			])

	def __len__(self):
		return len(self.imgs_list)

	def load_img(self, img_name):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		image_path = os.path.join(self.dataset_path, 'sg_dataset', f'sg_{self.image_set}_images',
								  img_name)
		assert os.path.exists(image_path), \
			'Path does not exist: {}'.format(image_path)
		img = Image.open(image_path).convert('RGB')
		return img

	def load_annotation(self, index):
		"""
		Load image and bounding boxes info from XML file in the PASCAL VOC
		format.
		"""
		sbj_boxes = []
		obj_boxes = []
		prd_boxes = []
		sbj_labels = []
		obj_labels = []
		prd_labels = []
		annotation = self.annotations[index]
		for spo in annotation:
			gt_sbj_label = spo['subject']['category']
			gt_sbj_bbox = spo['subject']['bbox']
			gt_obj_label = spo['object']['category']
			gt_obj_bbox = spo['object']['bbox']
			predicate = spo['predicate']

			# prepare bboxes for subject and object
			gt_sbj_bbox = y1y2x1x2_to_x1y1x2y2(gt_sbj_bbox)
			gt_obj_bbox = y1y2x1x2_to_x1y1x2y2(gt_obj_bbox)
			gt_pred_bbox = boxes_union(gt_sbj_bbox, gt_obj_bbox)
			
			# prepare labels for subject and object
			# map to word
			gt_sbj_label = self.all_objects[gt_sbj_label]
			gt_obj_label = self.all_objects[gt_obj_label]
			predicate = self.predicates[predicate]
			# map to new index
			gt_sbj_label = self._class_to_ind[gt_sbj_label]  
			gt_obj_label = self._class_to_ind[gt_obj_label] 			
			predicate = self._preds_to_ind[predicate]
			
			# append to list
			sbj_boxes.append(gt_sbj_bbox)
			obj_boxes.append(gt_obj_bbox)
			prd_boxes.append(gt_pred_bbox)
			sbj_labels.append(gt_sbj_label)
			obj_labels.append(gt_obj_label)
			prd_labels.append(predicate)

		sbj_boxes = torch.stack(sbj_boxes).type(torch.FloatTensor)
		obj_boxes = torch.stack(obj_boxes).type(torch.FloatTensor)
		prd_boxes = torch.stack(prd_boxes).type(torch.FloatTensor)
		sbj_labels = torch.tensor(sbj_labels, dtype=torch.int64)
		obj_labels = torch.tensor(obj_labels, dtype=torch.int64)
		prd_labels = torch.tensor(prd_labels, dtype=torch.int64)

		targets = {"sbj_boxes": sbj_boxes, "prd_boxes": prd_boxes, "obj_boxes": obj_boxes, \
		"sbj_labels": sbj_labels, "prd_labels": prd_labels, "obj_labels": obj_labels}

		return targets

	def __getitem__(self, index):
		img_name = self.imgs_list[index]
		img = self.load_img(img_name)
		targets = self.load_annotation(img_name)
		img, targets = self.transform(img, targets)
		return img, targets   # img: 3xHxW, targets: dict


def build(image_set, args):
    root = Path(args.vrd_path)
    assert root.exists(), f'provided VRD path {root} does not exist'
  
    dataset = VRDDataset(root, image_set)
    return dataset