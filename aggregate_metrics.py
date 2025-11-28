import argparse
import os
import pandas as pd
import numpy as np

LOCO_CLASSES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors"
]

def aggregate_results(args):
    if not os.path.exists(args.quantitative_folder):
        print(f"Error: Folder '{args.quantitative_folder}' does not exist.")
        return

    dfs = []
    missing_classes = []

    for class_name in LOCO_CLASSES:
        filename = f"{args.students_blocks}_{args.label}_{class_name}_{args.epochs_no}ep_{args.batch_size}bs.csv"
        filepath = os.path.join(args.quantitative_folder, filename)

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, sep=',')
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            missing_classes.append(class_name)
            print(f"Warning: Result file for '{class_name}' not found ({filename})")

    if not dfs:
        print("\nNo result files found matching the criteria. Exiting.")
        return

    # Concatenate all found dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Calculate the Mean
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    mean_values = combined_df[numeric_cols].mean()
    
    # Create a DataFrame for the average row
    mean_df = pd.DataFrame(mean_values).T
    mean_df['class_name'] = 'AVERAGE'
    
    # Reorder columns to ensure class_name is first
    cols = ['class_name'] + [c for c in mean_df.columns if c != 'class_name']
    mean_df = mean_df[cols]
    combined_df = combined_df[cols]

    # Create the final table (All Classes + Average Row)
    final_df = pd.concat([combined_df, mean_df], ignore_index=True)
    
    # Print summary to console
    print_markdown_summary(mean_values)

    # Save the aggregated results to a new CSV file
    output_filename = f"{args.label}_AVERAGE_{args.epochs_no}ep_{args.batch_size}bs.csv"
    output_path = os.path.join(args.quantitative_folder, output_filename)
    final_df.to_csv(output_path, index=False)

def print_markdown_summary(mean_vals):
    """Prints the average metrics in a formatted table similar to the inference script."""
    
    get_val = lambda k: mean_vals.get(k, 0.0)

    glob_auc = get_val('global_i_auroc')
    glob_30 = get_val('global_spro_30')
    glob_10 = get_val('global_spro_10')
    glob_05 = get_val('global_spro_05')
    glob_01 = get_val('global_spro_01')

    struct_auc = get_val('struct_i_auroc')
    struct_30 = get_val('struct_spro_30')
    struct_10 = get_val('struct_spro_10')
    struct_05 = get_val('struct_spro_05')
    struct_01 = get_val('struct_spro_01')

    logic_auc = get_val('logic_i_auroc')
    logic_30 = get_val('logic_spro_30')
    logic_10 = get_val('logic_spro_10')
    logic_05 = get_val('logic_spro_05')
    logic_01 = get_val('logic_spro_01')

    print("\n=== MEAN METRICS (All Classes) ===")
    header = f"{'Type':<12} | {'I-AUROC':<8} | {'sPRO@30%':<8} | {'sPRO@10%':<8} | {'sPRO@5%':<8} | {'sPRO@1%':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'Global':<12} | {glob_auc:.3f}    | {glob_30:.3f}    | {glob_10:.3f}    | {glob_05:.3f}    | {glob_01:.3f}")
    print(f"{'Structural':<12} | {struct_auc:.3f}    | {struct_30:.3f}    | {struct_10:.3f}    | {struct_05:.3f}    | {struct_01:.3f}")
    print(f"{'Logical':<12} | {logic_auc:.3f}    | {logic_30:.3f}    | {logic_10:.3f}    | {logic_05:.3f}    | {logic_01:.3f}")
    print("-" * len(header))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate MVTec LOCO Inference Results')
    
    parser.add_argument('--quantitative_folder', default='./results/loco/quantitatives_loco', type=str,
                        help='Path to the folder containing the CSV results.')
    
    parser.add_argument('--epochs_no', default=50, type=int,
                        help='Number of epochs used in the experiment filename.')
    
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size used in the experiment filename.')
    
    parser.add_argument('--students_blocks', type=str, default='Both_ViT', choices=['Both_ViT', 'Both_LLM', 'Full'],
                        help='Inference scenario.')
    
    parser.add_argument('--label', default='vit_norm_0.4', type=str,
                        help='The experiment label used during inference.')

    args = parser.parse_args()
    aggregate_results(args)