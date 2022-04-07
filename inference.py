# -*- coding: utf-8 -*-
"""detr_demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb

# Object Detection with DETR - a minimal implementation

In this notebook we show a demo of DETR (Detection Transformer), with slight differences with the baseline model in the paper.

We show how to define the model, load pretrained weights and visualize bounding box and class predictions.

Let's start with some common imports.
"""

# Commented out IPython magic to ensure Python compatibility.
from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import json
import argparse
torch.set_grad_enabled(False);

"""## DETR
Here is a minimal implementation of DETR:
"""

from main_vrd import get_args_parser
from models import build_model

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

model, criterion, postprocessors = build_model(args)
model.to("cpu")
model.eval()

model.load_state_dict(torch.load("/media/pranoy/Pranoy/detd_vrd_model.pth", map_location=torch.device('cpu')))

    # print(model)

    # x = torch.Tensor(2,3,224,224)
    # outputs = model(x)
    # print(outputs.keys())
    # print(outputs['pred_logits'][0, :, :-1].shape)
#     Demo DETR implementation.

#     Demo implementation of DETR in minimal number of lines, with the
#     following differences wrt DETR in the paper:
#     * learned positional encoding (instead of sine)
#     * positional encoding is passed at input (instead of attention)
#     * fc bbox predictor (instead of MLP)
#     The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
#     Only batch size 1 supported.
#     """
#     def __init__(self, num_classes, hidden_dim=256, nheads=8,
#                  num_encoder_layers=6, num_decoder_layers=6):
#         super().__init__()

#         # create ResNet-50 backbone
#         self.backbone = resnet50()
#         del self.backbone.fc

#         # create conversion layer
#         self.conv = nn.Conv2d(2048, hidden_dim, 1)

#         # create a default PyTorch transformer
#         self.transformer = nn.Transformer(
#             hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

#         # prediction heads, one extra class for predicting non-empty slots
#         # note that in baseline DETR linear_bbox layer is 3-layer MLP
#         self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
#         self.linear_bbox = nn.Linear(hidden_dim, 4)

#         # output positional encodings (object queries)
#         self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

#         # spatial positional encodings
#         # note that in baseline DETR we use sine positional encodings
#         self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
#         self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

#     def forward(self, inputs):
#         # propagate inputs through ResNet-50 up to avg-pool layer
#         x = self.backbone.conv1(inputs)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)

#         # convert from 2048 to 256 feature planes for the transformer
#         h = self.conv(x)

#         # construct positional encodings
#         H, W = h.shape[-2:]
#         pos = torch.cat([
#             self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
#             self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
#         ], dim=-1).flatten(0, 1).unsqueeze(1)

#         # propagate through the transformer
#         h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
#                              self.query_pos.unsqueeze(1)).transpose(0, 1)
        
#         # finally project transformer outputs to class labels and bounding boxes
#         return {'pred_logits': self.linear_class(h), 
#                 'pred_boxes': self.linear_bbox(h).sigmoid()}

"""As you can see, DETR architecture is very simple, thanks to the representational power of the Transformer. There are two main components:
* a convolutional backbone - we use ResNet-50 in this demo
* a Transformer - we use the default PyTorch nn.Transformer

Let's construct the model with 80 COCO output classes + 1 ⦰ "no object" class and load the pretrained weights.
The weights are saved in half precision to save bandwidth without hurting model accuracy.
"""

# detr = DETRdemo(num_classes=91)
# state_dict = torch.hub.load_state_dict_from_url(
#     url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
#     map_location='cpu', check_hash=True)
# detr.load_state_dict(state_dict)
# detr.eval();

"""## Computing predictions with DETR

The pre-trained DETR model that we have just loaded has been trained on the 80 COCO classes, with class indices ranging from 1 to 90 (that's why we considered 91 classes in the model construction).
In the following cells, we define the mapping from class indices to names.
"""

# COCO classes
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

import os 

with open(os.path.join(args.vrd_path, 'json_dataset', 'objects.json'), 'r') as f:
	CLASSES = json.load(f)


with open(os.path.join(args.vrd_path, 'json_dataset', 'predicates.json'), 'r') as f:
	PRD_CLASSES = json.load(f)

# root = os.path.join(self.dataset_path, 'sg_dataset', f'sg_{self.image_set}_images')

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

"""DETR uses standard ImageNet normalization, and output boxes in relative image coordinates in $[x_{\text{center}}, y_{\text{center}}, w, h]$ format, where $[x_{\text{center}}, y_{\text{center}}]$ is the predicted center of the bounding box, and $w, h$ its width and height. Because the coordinates are relative to the image dimension and lies between $[0, 1]$, we convert predictions to absolute image coordinates and $[x_0, y_0, x_1, y_1]$ format for visualization purposes."""

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

"""Let's put everything together in a `detect` function:"""

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas_prd = outputs['prd_logits'].softmax(-1)[0, :, :-1]
    keep = probas_prd.max(-1).values > 0.5
    bboxes_scaled_prd = rescale_bboxes(outputs['prd_boxes'][0, keep], im.size)


    # keep only predictions with 0.7+ confidence
    probas_sbj = outputs['sbj_logits'].softmax(-1)[0, :, :-1]
    # keep_s = probas_sbj.max(-1).values > 0.5
    # convert boxes from [0; 1] to image scales
    bboxes_scaled_sbj = rescale_bboxes(outputs['sbj_boxes'][0, keep], im.size)

  

    # keep only predictions with 0.7+ confidence
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    # keep_o = probas_obj.max(-1).values > 0.5
    bboxes_scaled_obj = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)
   

    print(len(probas_sbj[keep]))
    print(len(probas_prd[keep]))
    print(len(probas_obj[keep]))

    return probas_sbj[keep], probas_prd[keep], probas_obj[keep], bboxes_scaled_sbj, bboxes_scaled_prd, bboxes_scaled_obj

"""## Using DETR
To try DETRdemo model on your own image just change the URL below.
"""

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open("/Users/pranoyr/Desktop/vrd_sample/couple-people-man-girl.jpg")

scores_sbj, scores_prd, scores_obj, boxes_sbj, boxes_prd, boxes_obj = detect(im, model, transform)

"""Let's now visualize the model predictions"""

def plot_results(pil_img, scores_sbj, scores_prd, scores_obj, boxes_sbj, boxes_prd, boxes_obj):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    

    for s,p,o, (xmin, ymin, xmax, ymax), c in zip(scores_sbj, scores_prd, scores_obj, boxes_obj.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl_s = s.argmax()
        cl_p = p.argmax()
        cl_o = o.argmax()
        # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        text = f'{CLASSES[cl_s]}  {PRD_CLASSES[cl_p]} {CLASSES[cl_o]}'
        print(text)
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
plot_results(im, scores_sbj, scores_prd, scores_obj, boxes_sbj, boxes_prd, boxes_obj)