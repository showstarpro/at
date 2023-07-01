batch_size=8
save_dir='../data/model'
dataset='imagenet'
dataset_path='../data/dataset'
target_file='../data/at/val_rs.csv'
output_dir=''
surrogate_models=''
model_path='../data/model'
target_model=''
target_model_path='../data/model'
loss='ce'
device='cuda'

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

python train_weight.py ${training_start_params}