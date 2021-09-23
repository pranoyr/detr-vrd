import torch

def _get_src_permutation_idx(indices):
	# permute predictions following indices
	batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
	src_idx = torch.cat([src for (src, _) in indices])
	return batch_idx, src_idx



# indices = [(1,3), (2,2), (3,1)]
# indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# print(indices)
# print(_get_src_permutation_idx(indices))


# s = torch.Tensor(2,3)
# print(s)
# print(s[:,torch.tensor([1,1,1,1])])

# import torch.nn as nn

# class_embed = nn.Linear(10, 20)

# x = torch.randn(4,3,10, 10)
# print(class_embed(x).shape)



# CLASSES = [
#     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#     'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#     'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#     'toothbrush'
# ]

l = [torch.tensor(0.4), torch.tensor(0.3), torch.tensor(0.2), torch.tensor(0.1)]
print(sum(l))

target = {"boxes":[1,2,34]}
if "boxes" in target:	
	print("yes")

