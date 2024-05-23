import pandas as pd
import json
import os


if __name__ == '__main__':
    src = '/project/lt200203-aimedi/ipu24/raw/'
    coco = '/project/lt200203-aimedi/coco/images'
    coco_eng_train = json.load(open('/mnt/d/work/coco/annotations/captions_train2017.json'))
    coco_eng_val = json.load(open('/mnt/d/work/coco/annotations/captions_val2017.json'))
    coco_03 = json.load(open('/mnt/d/work/capgen/capgen_v0.3/annotations/capgen_v0.3_coco.json'))
    train_03 = json.load(open('/mnt/d/work/capgen/capgen_v0.3/annotations/capgen_v0.3_train.json'))
    test_03 = json.load(open('/mnt/d/work/capgen/capgen_v0.3/annotations/capgen_v0.3_test.json'))
    val_03 = json.load(open('/mnt/d/work/capgen/capgen_v0.3/annotations/capgen_v0.3_val.json'))
    train = {'url': [], 'caption': [], 'language': []}
    val = {'url': [], 'caption': [], 'language': []}
    test = {'url': [], 'caption': [], 'language': []}
    coco_041 = {}
    for k, v in coco_03.items():
        if k.startswith('test'):
            for caption in v:
                test['url'].append(os.path.join(src, k+'.jpg'))
                test['caption'].append(caption)
                test['language'].append('th')
        if k.startswith('train'):
            for caption in v:
                train['url'].append(os.path.join(src, k+'.jpg'))
                train['caption'].append(caption)
                train['language'].append('th')
        if k.startswith('val'):
            for caption in v:
                val['url'].append(os.path.join(src, k+'.jpg'))
                val['caption'].append(caption)
                val['language'].append('th')

    for k, v in train_03.items():
        for caption in v:
            train['url'].append(os.path.join(src, k))
            train['caption'].append(caption)
            train['language'].append('th')
    for k, v in val_03.items():
        for caption in v:
            val['url'].append(os.path.join(src, k))
            val['caption'].append(caption)
            val['language'].append('th')
    for k, v in test_03.items():
        for caption in v:
            test['url'].append(os.path.join(src, k))
            test['caption'].append(caption)
            test['language'].append('th')

    for ann in coco_eng_train['annotations']:
        image = f'{ann["image_id"]:012d}.jpg'
        train['url'].append(os.path.join(coco, 'train2017', image))
        train['caption'].append(ann['caption'])
        train['language'].append('en')
    for ann in coco_eng_val['annotations']:
        image = f'{ann["image_id"]:012d}.jpg'
        val['url'].append(os.path.join(coco, 'val2017', image))
        val['caption'].append(ann['caption'])
        val['language'].append('en')

    train_df = pd.DataFrame(train)
    train_df.to_csv('data/train.csv', index=False, sep='\t')
    val_df = pd.DataFrame(val)
    val_df.to_csv('data/val.csv', index=False, sep='\t')
    test_df = pd.DataFrame(test)
    test_df.to_csv('data/test.csv', index=False, sep='\t')
