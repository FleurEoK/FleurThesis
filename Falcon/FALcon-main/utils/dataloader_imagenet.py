import torch.utils.data as data
import torch
import torchvision
from PIL import Image
import os
import os.path
import numpy as np
import json
from torchvision.transforms import functional as F
import warnings
import random
import math
import copy
import numbers
import xml.etree.ElementTree as ET


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
    return pil_loader(path)

class ResizedImgAndBBox(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        #resize to 256
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                img = copy.deepcopy(img)
                ow, oh = w, h
            if w < h:
                ow = size
                oh = int(size*h/w)
            else:
                oh = size
                ow = int(size*w/h)
        else:
            ow, oh = size[::-1]
            w, h = img.size


        intersec = copy.deepcopy(bbox)
        ratew = ow / w
        rateh = oh / h

        intersec[:, 0] = bbox[:, 0]*ratew
        intersec[:, 2] = bbox[:, 2]*ratew
        intersec[:, 1] = bbox[:, 1]*rateh
        intersec[:, 3] = bbox[:, 3]*rateh


        #intersec = normalize_intersec(i, j, h, w, intersec)
        return (oh, ow), intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        size, crop_bbox = self.get_params(img, bbox, self.size)
        return F.resize(img, self.size, self.interpolation), crop_bbox



class RandomHorizontalFlipImgAndBBox(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bbox):
        if random.random() < self.p:
            x_bound, _ = img.size
            # assumption: box is 4-sized tuple of (x1, y1, x2, y2) format            
            flipbox = copy.deepcopy(bbox)
            flipbox[:, 0] = x_bound - bbox[:, 2]
            flipbox[:, 2] = x_bound - bbox[:, 0]
            return F.hflip(img), flipbox
        return img, bbox


# def load_train_bbox(label_dict, bbox_dir):
#     #bbox_dir = 'ImageNet/Projection/VGG16-448'
#     final_dict = {}
#     for i in range(len(label_dict)):    #use actual class count
#         now_name = label_dict[i]
#         now_json_file = os.path.join(bbox_dir, now_name + "_bbox.json")

#         with open(now_json_file, 'r') as fp:
#             name_bbox_dict = json.load(fp)

#         final_dict[i] = name_bbox_dict
#     return final_dict


def load_train_bbox(label_dict, bbox_dir, data_root):
    """
    Build bbox_dict[class_idx][absolute_img_path] → list_of_boxes.
    Missing or unreadable JSONs are tolerated.
    """
    final_dict = {}

    for cls_idx, wnid in label_dict.items():   # iterate over *all* classes
        json_path = os.path.join(bbox_dir, f"{wnid}_bbox.json")
        img_dir   = os.path.join(data_root, "train", wnid)   # <‑‑ NEW

        boxes_in_class = {}          # will hold absolute‑path keys

        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as fp:
                    raw = json.load(fp)       # { "n01443537_19475.JPEG": [...] , … }
            except (json.JSONDecodeError, OSError):
                raw = {}
        else:
            raw = {}

        # ----------  unify key format  ----------
        for old_key, gt_boxes in raw.items():
            # print('FILE NAME = ', fname)
            # ensure the JSON key has the extension
            # resolves theissue of the json being loaded from a different root location and joins the path correctly
            if not (old_key.lower().endswith(".jpeg") or old_key.lower().endswith(".jpg")):
                old_key += ".JPEG"
            fname = os.path.basename(old_key)
            abs_path = os.path.join(img_dir, fname)
            boxes_in_class[abs_path] = gt_boxes   # may be []
        # ----------------------------------------

        final_dict[cls_idx] = boxes_in_class      # even if empty

    return final_dict



def load_test_bbox(label_dict, all_imgs, bbox_dir):
    #bbox_dir = 'root/anno_val'
    locs = [(x[0].split('/')[-1],x[0],x[1]) for x in all_imgs]
    locs.sort()
    # example:
    # locs[0] = ('ILSVRC2012_val_00000293.JPEG',
    #            '/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012/val/n01440764/ILSVRC2012_val_00000293.JPEG',
    #            0)
    final_bbox_dict = {}
    for i in range(len(locs)):
        val_img_loc     = locs[i][1] # '/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012/val/n01440764/ILSVRC2012_val_00000293.JPEG'
        now_file_name   = locs[i][0].split('.')[0]      # 'ILSVRC2012_val_00000293',
        now_label       = label_dict[locs[i][2]]        # 'n01440764'
        now_xml_file    = os.path.join(bbox_dir, now_label, now_file_name + '.xml')
        gt_boxes        = get_cls_gt_boxes(now_xml_file, now_label)   
        
        final_bbox_dict[val_img_loc] = gt_boxes
    return final_bbox_dict
    

    
def load_val_bbox(label_dict,all_imgs,gt_location):
    #gt_location ='/data/zhangcl/DDT-code/ImageNet_gt'
    import scipy.io as sio
    gt_label = sio.loadmat(os.path.join(gt_location,'cache_groundtruth.mat'))
    locs = [(x[0].split('/')[-1],x[0],x[1]) for x in all_imgs]
    locs.sort()
    final_bbox_dict = {}
    for i in range(len(locs)):
        #gt_label['rec'][:,1][0][0][0], if multilabel then get length, for final eval
        final_bbox_dict[locs[i][1]] = gt_label['rec'][:,i][0][0][0][0][1][0]
    return final_bbox_dict


def get_cls_gt_boxes(xmlfile: str, cls: str):
    """Return a list of GT boxes (x1, y1, x2, y2) whose <name> == *cls*.

    *New in this revision*  
    – If *xmlfile* is absent **or** cannot be parsed, the function
      returns an empty list instead of raising, so the dataloader
      continues without killing the run.
    """
    # ── 1. Fast‑exit if the file isn’t there ────────────────────────────
    if not os.path.isfile(xmlfile):
        return []                  # skip this image

    # ── 2. Parse XML, but swallow malformed files gracefully ───────────
    try:
        tree = ET.parse(xmlfile)
    except (ET.ParseError, OSError):
        return []                  # corrupted / unreadable → skip

    # ── 3. Original extraction logic (unchanged) ───────────────────────
    gt_boxes = []
    for obj in tree.findall("object"):
        cls_name = obj.find("name").text
        if cls_name != cls:
            continue

        bbox = obj.find("bndbox")
        x1 = float(bbox.find("xmin").text)
        y1 = float(bbox.find("ymin").text)
        x2 = float(bbox.find("xmax").text)
        y2 = float(bbox.find("ymax").text)
        gt_boxes.append((x1, y1, x2, y2))

    if len(gt_boxes)==0:
        pass
        #print('%s bbox = 0'%cls)
    return gt_boxes


def load_all_test_bbox(label_dict, all_imgs, bbox_dir):
    """
    Build a dict that maps each validation‑image path to its ground‑truth
    boxes and labels.

    New in this revision
    --------------------
    • If the expected XML file is *absent* or *cannot be parsed*,
      the image is still entered in the output dict but receives
      empty lists for "gt_boxes" and "gt_labels" instead of
      aborting the whole run.
    """
    # ‑‑ original pre‑processing ‑‑
    locs = [(x[0].split('/')[-1], x[0], x[1]) for x in all_imgs.imgs]
    locs.sort()

    final_bbox_dict = {}

    for _, val_img_loc, cls_idx in locs:
        now_file_name = os.path.splitext(os.path.basename(val_img_loc))[0]
        now_label     = label_dict[cls_idx]
        now_xml_file  = os.path.join(bbox_dir, now_label, now_file_name + ".xml")

        gt_boxes, gt_labels = [], []

        # ‑‑ new robustness layer ‑‑
        if os.path.isfile(now_xml_file):
            try:
                tree  = ET.parse(now_xml_file)
                objs  = tree.findall("object")

                for obj in objs:
                    cls_name = obj.find("name").text
                    if cls_name not in all_imgs.class_to_idx:
                        continue

                    bnd = obj.find("bndbox")
                    x1 = float(bnd.find("xmin").text)
                    y1 = float(bnd.find("ymin").text)
                    x2 = float(bnd.find("xmax").text)
                    y2 = float(bnd.find("ymax").text)

                    gt_boxes .append((x1, y1, x2, y2))
                    gt_labels.append(all_imgs.class_to_idx[cls_name])

            except (ET.ParseError, OSError):
                # malformed or unreadable XML → treat as “no boxes”
                pass

        # Always register the image, even if lists are empty
        final_bbox_dict[val_img_loc] = {
            "gt_boxes" : gt_boxes,
            "gt_labels": gt_labels,
        }

    return final_bbox_dict


# def load_all_test_bbox(label_dict, all_imgs, bbox_dir):
#     #bbox_dir = 'root/anno_val'
#     locs = [(x[0].split('/')[-1],x[0],x[1]) for x in all_imgs.imgs]
#     locs.sort()
#     # example:
#     # locs[0] = ('ILSVRC2012_val_00000293.JPEG',
#     #            '/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012/val/n01440764/ILSVRC2012_val_00000293.JPEG',
#     #            0)
#     final_bbox_dict = {}
#     for i in range(len(locs)):
#         val_img_loc     = locs[i][1] # '/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012/val/n01440764/ILSVRC2012_val_00000293.JPEG'
#         now_file_name   = locs[i][0].split('.')[0]      # 'ILSVRC2012_val_00000293',
#         now_label       = label_dict[locs[i][2]]        # 'n01440764'
#         now_xml_file    = os.path.join(bbox_dir, now_label, now_file_name + '.xml')
        
#         tree = ET.parse(now_xml_file)
#         objs = tree.findall('object')
#         gt_boxes    = []
#         gt_labels   = []
#         for obj in objs:
#             bbox        = obj.find('bndbox')
#             cls_name    = obj.find('name').text
#             if not (cls_name in all_imgs.class_to_idx.keys()):
#                 continue
#             x1 = float(bbox.find('xmin').text)#-1
#             y1 = float(bbox.find('ymin').text)#-1
#             x2 = float(bbox.find('xmax').text)#-1
#             y2 = float(bbox.find('ymax').text)#-1
            
#             gt_boxes.append((x1, y1, x2, y2))
#             gt_labels.append(all_imgs.class_to_idx[cls_name])
#         final_bbox_dict[val_img_loc] = {"gt_boxes": gt_boxes, "gt_labels": gt_labels}
#     return final_bbox_dict







class ImageNetDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, pseudo_bbox_dir, gt_bbox_dir, input_size=256, train=True, transform=None, target_transform=None, loader=default_loader):
        from torchvision.datasets import ImageFolder
        self.train              = train
        self.input_size         = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.pseudo_bbox_dir    = pseudo_bbox_dir
        self.gt_bbox_dir        = gt_bbox_dir
        
        if self.train:
            self.img_dataset = ImageFolder(os.path.join(root, 'train'))
        else:
            self.img_dataset = ImageFolder(os.path.join(root, 'val'))
        
        if len(self.img_dataset) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.label_class_dict = {}
        for k, v in self.img_dataset.class_to_idx.items():
            self.label_class_dict[v] = k


        if self.train:
            #load train bbox
            self.bbox_dict = load_train_bbox(self.label_class_dict, self.pseudo_bbox_dir, root)
        else:
            #load test bbox
            self.bbox_dict = load_test_bbox(self.label_class_dict, self.img_dataset.imgs, self.gt_bbox_dir)
            self.bbox_dict_all = load_all_test_bbox(self.label_class_dict, self.img_dataset, self.gt_bbox_dir)
            # self.bbox_dict = load_val_bbox(self.label_class_dict, self.img_dataset.imgs, self.gt_bbox_dir)

        self.img_dataset        = self.img_dataset.imgs
        self.transform          = transform
        self.target_transform   = target_transform
        self.loader             = loader
        self.fetch_one_bbox     = True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path, target = self.img_dataset[index]
        print('path: ', path)
        print('target: ', target)
        img = self.loader(path)
        
        if self.train:
            bbox = self.bbox_dict[target][path]
        else:
            if self.fetch_one_bbox:
                bbox = self.bbox_dict[path]
            else:
                objects     = self.bbox_dict_all[path]
                bbox        = objects["gt_boxes"]
                bbox_labels = objects["gt_labels"]
        
        w, h = img.size
        bbox = np.array(bbox, dtype='float32')

        # ADD DEBUGGING HERE (after line 409):
        if bbox.size == 0:
            print(f"DEBUG Sample {index}: Empty bbox!")
            print(f"  Path: {path}")
            print(f"  Target class idx: {target}")
            print(f"  Target class name: {self.label_class_dict[target]}")
            print(f"  Image size: {w}x{h}")
            print(f"  bbox before np.array: {self.bbox_dict[path] if not self.train else self.bbox_dict[target][path]}")
        

        # Ensure bbox is always 2D, even for single boxes
        if bbox.ndim == 1:
            bbox = bbox.reshape(1, -1)
        
        # Preparing bounding boxes
        if self.train:
            # During training, we expect bbox to be in (x,y,w,h) format, normalized to (0,1) range            
            # convert from (x, y, w, h) to (x1,y1,x2,y2) so that they can be used with ResizedImgAndBBox and RandomHorizontalFlipImgAndBBox
            bbox[0] = bbox[0]
            # bbox[2] = bbox[0] + bbox[2]
            # bbox[1] = bbox[1]
            # bbox[3] = bbox[1] + bbox[3]
            # bbox[0] = math.ceil(bbox[0] * w)
            # bbox[2] = math.ceil(bbox[2] * w)
            # bbox[1] = math.ceil(bbox[1] * h)
            # bbox[3] = math.ceil(bbox[3] * h)
            # bbox = np.expand_dims(bbox, 0)

            if bbox.ndim == 1:
                # Single bbox as 1D array
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                bbox[0] = math.ceil(bbox[0] * w)
                bbox[2] = math.ceil(bbox[2] * w)
                bbox[1] = math.ceil(bbox[1] * h)
                bbox[3] = math.ceil(bbox[3] * h)
                bbox = np.expand_dims(bbox, 0)
            else:
                # Multiple bboxes as 2D array (N, 4)
                bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
                bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
                bbox[:, 0] = np.ceil(bbox[:, 0] * w)
                bbox[:, 2] = np.ceil(bbox[:, 2] * w)
                bbox[:, 1] = np.ceil(bbox[:, 1] * h)
                bbox[:, 3] = np.ceil(bbox[:, 3] * h)

            img_i, bbox_i   = ResizedImgAndBBox(self.input_size)(img, bbox)
            img, bbox       = RandomHorizontalFlipImgAndBBox()(img_i, bbox_i)
        else:
            # During test, we expect bbox to be in (x1, y1, x2, y2) format already, 
            # but, not normalized: to be in range of image dimensions (width and height)
            img, bbox       = ResizedImgAndBBox(self.input_size)(img, bbox)

        #convert from (x1, y1, x2, y2) back to (x, y, w, h)
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
        bbox = torch.tensor(bbox)
        
        if self.fetch_one_bbox:
            bbox_areas = bbox[:, 2]*bbox[:, 3]
            largest_bbox_id = bbox_areas.argmax().item()
            bbox = bbox[largest_bbox_id, :]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train: # image, image label (one), and pseudo bounding box (one)
            return img, target, bbox 
        else:
            if self.fetch_one_bbox: # image, image label (one), and ground truth bounding box (one)
                return img, target, bbox
            else: # image, bounding box labels (one for each bounding box), and ground truth bounding boxes (multiple)
                return img, bbox_labels, bbox

        

    def __len__(self):
        return len(self.img_dataset)


if __name__ == '__main__':
    a =ImageNetDataset('$HOME/imagenet-subset/ILSVRC/Data/CLS-LOC/', train=True)
