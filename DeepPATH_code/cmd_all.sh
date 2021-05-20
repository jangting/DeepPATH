#!/bin/bash

prepro_image_dir_path="outputs/slides"
CODE_DIR="/home/jangting/sources/DeepPATH/DeepPATH_code"

### 0.1 Tile the svs slide images
#python 00_preprocessing/0b_tileLoop_deepzoom4.py -s 299 -e 0 -j 32 -B 25 -o ${prepro_image_dir_path} "/home/jangting/dataset/TCGA-LUNG/data/*/*svs"

### 0.2a Sort the tiles into train/valid/test sets according to the classes defined
#python 00_preprocessing/0d_SortTiles.py --SourceFolder=${prepro_image_dir_path} --JsonFile=./example_TCGA_lung/metadata.cart.2017-03-02T00_36_30.276824.json --Magnification=20.0 --MagDiffAllowed=0 --SortingOption=3 --PercentTest=15 --PercentValid=15 --PatientID=12 --nSplit 0 >> log_sort_tiles.log

# (PASS) 0.2b Vahadane normalization
# split -l 200 img_list.txt splitted_img_list

### 0.3a Convert the JPEG tiles into TFRecord format for 2 or 3 classes jobs
#mkdir jpeg_label_dir
#mv Solid_Tissue_Normal TCGA-LU* jpeg_label_dir

#python 00_preprocessing/TFRecord_2or3_Classes/build_image_data.py --directory='jpeg_label_dir' --output_directory='outputs/tfrecord' --train_shards=1024  --validation_shards=128 --num_threads=4
#python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='jpeg_tile_dir'  --output_directory='outputs/tfrecord' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test'
#python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='jpeg_label_dir'  --output_directory='outputs/tfrecord' build_TF_test --one_FT_per_Tile=False --ImageSet_basename='valid'


### 1.1.a Build the model

## install bazel
#curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
#sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
#echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

#sudo apt update
#sudo apt install bazel

cd ${CODE_DIR}/01_training/xClasses \
    && rm bazel-* \
    && bazel build inception/imagenet_train


### 1.1.b train the model (PASS)


### 1.2 - Transfer learning

## 1.2.1 Download 'inceptionV3' pretrained model
#cd ${CODE_DIR}
#curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
#tar -xvf inception-v3-2016-03-01.tar.gz

#mkdir -p ${CODE_DIR}/output_model_transfer

## 1.2.2 transfer learning
max_steps=100
model='inception-v3'
#model='inception-v4'

cd ${CODE_DIR}/01_training/xClasses \
    && bazel-bin/inception/imagenet_train \
   --num_gpus=1 --batch_size=30 --train_dir=${CODE_DIR}/output_model_transfer_${max_steps}/ \
   --data_dir=${CODE_DIR}/outputs/tfrecord/ \
   --pretrained_model_checkpoint_path=${CODE_DIR}/${model}/model.ckpt-157585 \
   --fine_tune=True --initial_learning_rate=0.001 --ClassNumber=3 \
   --mode='0_softmax' --save_step_for_chekcpoint=100 --max_steps=${max_steps} \
   --model=${model}


### 1.3 Validation (FAIL)
#mkdir -p ${CODE_DIR}/output_eval_valid

#cd ${CODE_DIR}/02_testing/xClasses \
#    && python nc_imagenet_eval.py \
#  --checkpoint_dir=${CODE_DIR}/output_model_transfer/ \
#  --eval_dir=${CODE_DIR}/output_eval_valid \
#  --data_dir=${CODE_DIR}/outputs/tfrecord/ \
#  --batch_size 30 --ImageSet_basename='valid' --ClassNumber 3 \
#  --mode='0_softmax' --run_once --TVmode='valid'
