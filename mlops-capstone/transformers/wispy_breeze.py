import os
import json
import glob
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data_dir, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    annotations_file = transform_annotation(data_dir)
    test_file = create_annotation_files(data_dir, annotations_file)
    file_processing(data_dir)
    convert_COCO_to_Torch(data_dir, annotations_file)

    return data_dir

def transform_annotation(data_dir):
    source_annonation_path = f'{data_dir}/data/annotations.json'
    target_annotation_path = f'{data_dir}/annotations.json'

    if os.path.isfile(target_annotation_path):
        print("\nFile already exists")
    else:
        with open(source_annonation_path, 'r') as openfile:
        
            data = json.load(openfile)

            for i in data['images']:
                path = i.get("file_name")
                dir , file_name = path.split('/')
                name, _extension = file_name.split('.')
                file_name = f'{name}.jpg'
                _batch, num = dir.split('_')
                new_path = f'images/{num}{file_name}'
                i.update({"file_name" : new_path})
                path = i.get("file_name")
                
            with open(target_annotation_path, "w") as outfile:
                json.dump(data, outfile)
    return target_annotation_path

def create_annotation_files(data_dir, annotations_file):
    labels_dir = f'{data_dir}/labels'
    test_file_dir = f'{labels_dir}/15000084.jpg'

    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)
        
    if os.path.isfile(test_file_dir):
        print("Labels already exist")
    else:
        with open(annotations_file, 'r') as openfile:
        
            data = json.load(openfile)

            img_ids = {}
                
            for i in data['images']:
                path = i.get("file_name")
                id = i.get("id")
                name, ext = path.split('.')
                # file_name = f'{path}.txt'
                # f = open(file_name, 'a')
                # f.close()
                img_ids.update({id: name})
                
            for j in data['annotations']:
                id = j.get("id")
                img_id = j.get("image_id")
                class_id = j.get("category_id")
                segmentation = j.get("segmentation")
                images , target_file = img_ids.get(img_id).split('/')
                file_name = f'{labels_dir}/{target_file}.txt'
                segmentation_string = ' '.join(str(v) for v in segmentation[0])
                f = open(file_name, 'a')
                f.writelines(f'{class_id} {segmentation_string}')
                f.close()
    return test_file_dir

def file_processing(dataset_dir):
    images_path = f'{dataset_dir}/images'
    data_dir = f'{dataset_dir}/data'
    temp_dir = f'{data_dir}/temp'

    if os.path.exists(images_path):
        print("\nDataset directory exists.")
    else:
        try:
            os.mkdir(images_path)
        except OSError as error:
            print(error)
                
    if os.path.exists(temp_dir):
        print("\nDataset directory exists.")
    else:
        os.mkdir(temp_dir)
    
    print('\nStandardizing extensions')
    for old_name in glob.glob(f'{data_dir}/*/*.JPG'):
        path, file_name = old_name.split('data/')
        batch, name = file_name.split('/')
        ony_name, ext = name.split('.')
        temp_name = f'{temp_dir}/{ony_name}.jpg'
        shutil.copy(old_name, temp_name)
        os.remove(old_name)
        new_name = f'{path}data/{batch}/{ony_name}.jpg'
        shutil.copy(temp_name, new_name)
        os.remove(temp_name)
        is_true = os.path.isfile(f'{new_name}')
    
    if os.path.isfile(f'{images_path}/15000084.jpg'):
        print("\nFile already exists")
    else:
        print('\nCopying files')
        for source in glob.glob(f'{data_dir}/*/*.jpg'):
            _x , path = source.split('data/')
            batch , file_name = path.split('/')
            _y , num = batch.split('_')
            destination = f'{images_path}/{num}{file_name}'
            shutil.copyfile(source, destination)

def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s
    
def convert_COCO_to_Torch(output_path, annotations_file, use_segments=True):
    train_dir = f'{output_path}/train'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
      # make_folders(output_path)
    if os.path.isfile(f"{train_dir}/1000061.jpg"):
        print("Labels already exist")
    else:

        df_img_id = []
        df_img_name = []
        df_img_width = []
        df_img_height = []
        with open(annotations_file) as f:
            json_data = json.load(f)
        print('extract the json :', annotations_file)

        # write _darknet.labels, which holds names of all classes (one class per line)
        label_file = os.path.join(train_dir, "_darknet.labels.txt")
        with open(label_file, "w") as f:
            for image in json_data["images"]:
                img_id = image["id"]
                #img_name = image["file_name"]
                img_name = os.path.basename(image["file_name"])
                json_images = os.path.dirname(annotations_file)+"/images/"+img_name
                target = train_dir+"/"+img_name
                shutil.copyfile(json_images, target) 
                img_width = image["width"]
                img_height = image["height"]
                df_img_id.append(img_id)
                df_img_name.append(img_name)
                df_img_width.append(img_width)
                df_img_height.append(img_height)
            
                anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
                #anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
                anno_txt = os.path.join(train_dir, img_name.replace(".jpg",".txt"))

                h, w, f = image['height'], image['width'], image['file_name']
                bboxes = []
                segments = []
                with open(anno_txt, "w") as f:
                    for anno in anno_in_image:
                        category = anno["category_id"]
                        bbox_COCO = anno["bbox"]
                        #x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                        if anno['iscrowd']:
                            continue
                        # The COCO box format is [top left x, top left y, width, height]
                        box = np.array(anno['bbox'], dtype=np.float64)
                        box[:2] += box[2:] / 2  # xy top-left corner to center
                        box[[0, 2]] /= w  # normalize x
                        box[[1, 3]] /= h  # normalize y
                        if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                            continue
                        #cls = coco80[anno['category_id'] - 1] if cls91to80 else anno['category_id'] - 1  # class
                        cls = anno['category_id']
                        box = [cls] + box.tolist()
                        if box not in bboxes:
                            bboxes.append(box)
                        # Segments
                        if use_segments:
                            if len(anno['segmentation']) > 1:
                                s = merge_multi_segment(anno['segmentation'])
                                s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                            else:
                                s = [j for i in anno['segmentation'] for j in i]  # all segments concatenated
                                s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                            s = [cls] + s
                            if s not in segments:
                                segments.append(s)

                        last_iter=len(bboxes)-1
                        line = *(segments[last_iter] if use_segments else bboxes[last_iter]),  # cls, box or segments
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
        print("creating category_id and category name in darknet.labels")
        with open(label_file, "w") as f:
            for category in json_data["categories"]:
                category_name = category["name"]
                print(category_name)
                f.write(f"{category_name}\n")

        with open(f'{output_path}/manifest.json', "w") as f:
            today = datetime.now()
            todays_date = f'{today.year} {today.month} {today.day}'
            manifest = {"info": {"date": todays_date}, "images": []}
            for file in json_data["images"]:
                _dir , file_name = file["file_name"].split("/")
                manifest["images"] += [file_name]
            y = json.dump(manifest, f)
        print("Finish")


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
