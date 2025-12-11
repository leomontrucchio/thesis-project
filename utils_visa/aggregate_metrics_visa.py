import argparse
import os
import pandas as pd
import numpy as np

# VisA Dataset Classes
VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", 
    "fryum", "macaroni1", "macaroni2", "pcb1", 
    "pcb2", "pcb3", "pcb4", "pipe_fryum"
]

def aggregate_results(args):
    if not os.path.exists(args.quantitative_folder):
        print(f"Error: Folder '{args.quantitative_folder}' does not exist.")
        return

    dfs = []
    for class_name in VISA_CLASSES:
        filename = f"{args.label}_{class_name}_{args.epochs_no}ep_{args.batch_size}bs.csv"
        filepath = os.path.join(args.quantitative_folder, filename)

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, sep='&')
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"Warning: Result file for '{class_name}' not found.")

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
    
    # Reorder columns
    cols = ['class_name'] + [c for c in combined_df.columns if c != 'class_name']
    combined_df = combined_df[cols]
    mean_df = mean_df[cols]

    # Create the final table
    final_df = pd.concat([combined_df, mean_df], ignore_index=True)
    
    # Print summary to console
    print_markdown_summary(mean_values)

    # Save the aggregated results to a new CSV file
    output_filename = f"{args.label}_AVERAGE_{args.epochs_no}ep_{args.batch_size}bs.csv"
    output_path = os.path.join(args.quantitative_folder, output_filename)
    final_df.to_csv(output_path, index=False, sep=',')

def print_markdown_summary(mean_vals):
    """Prints the average metrics for all quartiles."""
    
    get_val = lambda k: mean_vals.get(k, 0.0)

    p_auc = get_val('p_auroc')
    i_auc = get_val('i_auroc')

    # Helper to fetch row values
    def get_quartile_row(q):
        return [get_val(f'aupro_30_{q}'), get_val(f'aupro_10_{q}'), get_val(f'aupro_05_{q}'), get_val(f'aupro_01_{q}')]

    q4 = get_quartile_row('q4')
    q3 = get_quartile_row('q3')
    q2 = get_quartile_row('q2')
    q1 = get_quartile_row('q1')

    print("\n" + "="*80)
    print(f"=== MEAN METRICS (VisA) ===")
    print("="*80)

    print(f"P-AUROC  |  I-AUROC")
    print(f"  {p_auc:.3f}   |   {i_auc:.3f}\n")

    header = f" {'QUARTILE':^8} | {'AUPRO@30%':^11} | {'AUPRO@10%':^11} | {'AUPRO@5%':^10} | {'AUPRO@1%':^10} |"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    print(f" {'Q4':^8} | {q4[0]:^11.3f} | {q4[1]:^11.3f} | {q4[2]:^10.3f} | {q4[3]:^10.3f} |")
    print(f" {'Q3':^8} | {q3[0]:^11.3f} | {q3[1]:^11.3f} | {q3[2]:^10.3f} | {q3[3]:^10.3f} |")
    print(f" {'Q2':^8} | {q2[0]:^11.3f} | {q2[1]:^11.3f} | {q2[2]:^10.3f} | {q2[3]:^10.3f} |")
    print(f" {'Q1':^8} | {q1[0]:^11.3f} | {q1[1]:^11.3f} | {q1[2]:^10.3f} | {q1[3]:^10.3f} |")
    print("-" * len(header))
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate VisA Inference Results')
    
    parser.add_argument('--quantitative_folder', default='./results/visa/quantitatives_visa', type=str,
                        help='Path to the folder containing the CSV results.')
    
    parser.add_argument('--epochs_no', default=50, type=int,
                        help='Number of epochs used in the experiment filename.')
    
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size used in the experiment filename.')
    
    parser.add_argument('--label', default='assistant_inspect_db_LLM_0_1', type=str,
                        help='The experiment label used during inference.')

    args = parser.parse_args()
    aggregate_results(args)