# V1 can load pretrained model of VSS Block

MAMBA_MODEL=$1
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset120_RoadSeg/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="data/nnUNet_results/Dataset120_RoadSeg/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="0,1"

# train
#CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 120 2d all -tr ${MAMBA_MODEL} -num&&

# predict
#echo "Predicting..." &&
#CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
#    -i "data/nnUNet_raw/Dataset120_RoadSeg/imagesTs" \
#    -o "${PRED_OUTPUT_PATH}" \
#    -d 120 \
#    -c 2d \
#    -tr "${MAMBA_MODEL}" \
#    --disable_tta \
#    -f all \
#    -chk "checkpoint_best.pth" &&

echo "Computing F1..."
python evaluation/compute_cell_metric.py \
    --gt_path "data/nnUNet_raw/Dataset120_RoadSeg/labelsTs" \
    -s "${PRED_OUTPUT_PATH}" \
    -o "${EVAL_METRIC_PATH}" \
    -n "${MAMBA_MODEL}_120_2d"  &&

echo "Done."