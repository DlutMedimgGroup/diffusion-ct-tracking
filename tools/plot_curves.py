# plot_curves.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_logs(log_dir: str) -> dict:
    """
    Loads all scalar data from a specified TensorBoard log directory.

    Args:
        log_dir (str): The path to the directory containing TensorBoard event files.

    Returns:
        dict: A dictionary where keys are the scalar tags (e.g., 'train/loss')
              and values are pandas DataFrames with 'epoch' and 'value' columns.
    """
    print(f"Loading TensorBoard logs from: '{log_dir}'...")
    
    # Initialize the EventAccumulator
    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={event_accumulator.SCALARS: 0} # 0 means load all scalar events
        )
        ea.Reload() # Load the events from disk
    except Exception as e:
        print(f"Error loading EventAccumulator: {e}")
        return {}

    # Extract scalar data into a dictionary of pandas DataFrames
    all_tags = ea.Tags().get('scalars', [])
    if not all_tags:
        print("No scalar data found in the specified directory.")
        return {}
        
    log_data = {}
    for tag in all_tags:
        events = ea.Scalars(tag)
        log_data[tag] = pd.DataFrame(
            [(event.step, event.value) for event in events],
            columns=['epoch', 'value']
        )
        print(f"  - Loaded tag '{tag}' with {len(log_data[tag])} data points.")
        
    return log_data

def plot_and_save_curves(log_data: dict, output_path: str, skip_epochs: int = 0):
    """
    Generates and saves a plot of the training curves.

    Args:
        log_data (dict): The dictionary of pandas DataFrames from `load_tensorboard_logs`.
        output_path (str): The full path to save the output PNG file.
        skip_epochs (int): The number of initial epochs to skip for plotting.
    """
    if not log_data:
        print("No data to plot.")
        return

    # Dynamically create subplots based on the number of tags found
    num_plots = len(log_data)
    # Arrange plots in a grid, e.g., max 4 columns
    ncols = min(num_plots, 4)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        figsize=(6 * ncols, 5 * nrows), 
        squeeze=False,
        tight_layout=True
    )
    axes = axes.flatten()

    # Plot each metric
    for i, (tag, df) in enumerate(log_data.items()):
        plot_df = df.iloc[skip_epochs:]
        
        # 如果切片后没有数据，则跳过此图
        if plot_df.empty:
            print(f"  - Skipping plot for '{tag}' as no data remains after skipping the first {skip_epochs} epochs.")
            axes[i].set_visible(False) # 隐藏这个空的子图
            continue
        
        ax = axes[i]
        # 使用切片后的 `plot_df` 进行绘图
        ax.plot(plot_df['epoch'], plot_df['value'], marker='o', linestyle='-', markersize=4)
        
        title = tag.title()
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    # Add a main title to the figure
    main_title = "Training & Validation Curves"
    if skip_epochs > 0:
        main_title += f" (Ignoring First {skip_epochs} Epochs)" # 在标题中注明
    fig.suptitle(main_title, fontsize=16, y=1.02)

    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close(fig)

if __name__ == "__main__":
    # ===================================================================
    #   CONFIGURATION
    # ===================================================================
    LOG_DIRECTORY = "/home/mingrui/disk1/projects/20260112_DiffusionCorr/projects/trained_models/AE_1mm_coarse_4"
    
    # --- 设置要跳过的 epoch 数量 ---
    EPOCHS_TO_SKIP = 5
    # -----------------------------------------

    # Construct the full, absolute path to the log directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    full_log_dir = os.path.join(project_root, LOG_DIRECTORY)
    
    # Define the output file path
    # 为了区分，可以在文件名中包含跳过的epoch数量
    output_filename = f"training_curves.png"
    output_plot_path = os.path.join(full_log_dir, output_filename)

    if not os.path.isdir(full_log_dir):
        print(f"Error: The specified log directory does not exist.")
        print(f"Checked path: {full_log_dir}")
    else:
        # 1. Load the data from TensorBoard logs
        data = load_tensorboard_logs(full_log_dir)
        
        # 2. Plot the data and save the figure, passing the new parameter
        plot_and_save_curves(data, output_plot_path, skip_epochs=EPOCHS_TO_SKIP)
