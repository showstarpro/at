batch_size=32
save_dir="../data/model"
dataset="imagenet"
dataset_path="/home/dataset/ImageNet/ILSVRC2012_img_val"
target_file="../data/at/val_rs.csv"
output_dir="../data"
surrogate_models="inception_v4 resnet18 densenet161 vgg16_bn"
model_path="../data/model"
target_model="resnet50"
target_model_path="../data/model"
loss="ce"
device="cuda"
sgpu=0



training_start_params=" \
--batch_size ${batch_size} \
--save_dir ${save_dir} \
--dataset ${dataset} \
--dataset_path ${dataset_path} \
--target_file ${target_file} \
--output_dir ${output_dir} \
--surrogate_models ${surrogate_models} \
--model_path ${model_path} \
--target_model ${target_model} \
--target_model_path ${target_model_path} \
--loss ${loss} \
--device ${device} \
--sgpu ${sgpu}
"

python train_grad.py ${training_start_params}