# Dataset Preparation

QueenVIS uses Detectron2's builtin dataset convention. All datasets are located through the `DETECTRON2_DATASETS` environment variable.

Under this directory, Detectron2 will look for datasets in the structure described below:

```text
$DETECTRON2_DATASETS/
  coco/
  ytvis_2019/
  ytvis_2021/
  ovis/
```

Set the location of the builtin datasets with:

```bash
export DETECTRON2_DATASETS=/path/to/datasets
```

If `DETECTRON2_DATASETS` is left unset, the default is `./datasets`, relative to the current working directory.

<!--
The model zoo contains configurations and models that use these builtin
datasets. Add the final model-zoo link here when it is available.
-->

## Step 1: Prepare image and video instance segmentation datasets

### COCO

Download [COCO 2017](https://cocodataset.org/#download) and arrange it as follows:

```text
coco/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # Image files referenced by the corresponding JSON annotation file
```

### YouTube-VIS 2019

Download [YouTube-VIS 2019](https://competitions.codalab.org/competitions/20128) and arrange it as follows:

```text
ytvis_2019/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

### YouTube-VIS 2021

Download [YouTube-VIS 2021](https://competitions.codalab.org/competitions/28988) and arrange it as follows:

```text
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

### OVIS

Download [OVIS](https://competitions.codalab.org/competitions/32377) and arrange it as follows:

```text
ovis/
  annotations/
    {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

## Step 2: Prepare annotations for combined training data

From the QueenVIS repository root, run:

```bash
python convert_coco2ytvis.py
```

This generates COCO annotations converted to the formats expected by the YouTube-VIS 2019, YouTube-VIS 2021, and OVIS training configurations.

### Expected final dataset structure

```text
$DETECTRON2_DATASETS/
+-- coco/
|   +-- annotations/
|   |   +-- instances_{train,val}2017.json
|   |   +-- coco2ytvis2019_train.json
|   |   +-- coco2ytvis2021_train.json
|   |   +-- coco2ovis_train.json
|   +-- {train,val}2017/
|       +-- *.jpg
|
+-- ytvis_2019/
|   ...
|
+-- ytvis_2021/
|   ...
|
+-- ovis/
    ...
```

Before training, verify that the paths referenced by the generated JSON files resolve relative to `$DETECTRON2_DATASETS`.
