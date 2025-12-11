export CUDA_VISIBLE_DEVICES=0

class_names=("breakfast_box" "juice_bottle" "pushpins" "screw_bag" "splicing_connectors")

for class_name in "${class_names[@]}"
    do
        python3 infer_loco_hd.py --class_name $class_name
    done

python3 ./utils_loco/aggregate_metrics_loco.py