source /home/miatang/miniconda3/etc/profile.d/conda.sh
conda activate /home/miatang/miniconda3/envs/gsa/
echo "Conda environment activated:" $CONDA_DEFAULT_ENV

cd /mnt/sda/miatang/external_git/Grounded-Segment-Anything
echo "Working directory changed to" $PWD

input_image=$1
output_dir=$2
echo "Running auto label with: " $input_image $output_dir 

export CUDA_VISIBLE_DEVICES=0
python automatic_label_ram_ours.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --ram_checkpoint ram_swin_large_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image $input_image \
  --output_dir $output_dir \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cuda"