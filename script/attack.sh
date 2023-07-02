attack_name='mi_fgsm'
eps=8
batch_size=8
save_dir="../data/model"
dataset="imagenet"
dataset_path="/home/dataset/ImageNet/ILSVRC2012_img_val"
target_file="../data/at/val_rs.csv"
output_dir="../data"
surrogate_models="inception_v4"
model_path="../data/model"
target_model="inception_v3"
target_model_path="../data/model"
loss="ce"
device="cuda"
sgpu=0

training_start_params=" \
--attack_name ${attack_name} \
--eps ${eps} \
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

python attack_weight.py ${training_start_params}