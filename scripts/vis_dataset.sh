# bash scripts/vis_dataset.sh

dataset_path=/home/ze/projects/Improved-3D-Diffusion-Policy/training_data_example

vis_cloud=0
cd Improved-3D-Diffusion-Policy
python vis_dataset.py --dataset_path $dataset_path \
                    --use_img 1 \
                    --vis_cloud ${vis_cloud} \
                    --use_pc_color 0 \
                    --downsample 1 \