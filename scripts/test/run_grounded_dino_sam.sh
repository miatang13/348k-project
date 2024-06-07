source /home/miatang/miniconda3/etc/profile.d/conda.sh
conda activate /home/miatang/miniconda3/envs/gsa/
echo "Conda environment activated:" $CONDA_DEFAULT_ENV

cd /mnt/sda/miatang/external_git/Grounded-Segment-Anything
echo "Working directory changed to" $PWD

export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "/home/miatang/projects/stroke-label/experiments/grounded_segmentation/test_outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "bear" \
  --device "cuda"