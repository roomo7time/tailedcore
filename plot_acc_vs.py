
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

label_col_idx=0
x_col_idx = 3
font_size = 16

base_df = pd.read_csv('./logs_plot/tailedpatch_result_summary - acc_vs.csv')
grouped = base_df.groupby(base_df.columns[label_col_idx])
dfs = {group: data for group, data in grouped}

sns.set_theme(style="whitegrid")
palette = sns.color_palette("viridis", n_colors=len(dfs))  # 'viridis' palette for n-1 lines]

for y_col_idx in [5,6,7,8,10,11, 16, 17]:
    plt.figure()
    for i, (_, df) in enumerate(dfs.items()):
        x_col_name = df.columns[x_col_idx]  # This gets the first column name for the x-axis label
        y_col_name = df.columns[y_col_idx]  # This gets the second column name for the y-axis label
        label_name = df.iloc[0, label_col_idx]  # Assuming the label you want is in the first row of the specified column
        sns.lineplot(data=df, x=x_col_name, y=y_col_name, marker='o', linestyle='-', color=palette[i], markersize=8, label=label_name) # with legend
        # sns.lineplot(data=df, x=x_col_name, y=y_col_name,  color=palette[i], marker='o', linestyle='-', markersize=8)

    plt.xlabel(x_col_name, size=font_size)  # Set the X axis label to the column name
    plt.ylabel(y_col_name, size=font_size)  # Set the Y axis label to the column name
    plt.tight_layout()  # Adjust the layout
    

    # Save the plot as a PDF file
    plt.savefig(f'line_{x_col_name.lower().replace(" ", "_")}_vs_{y_col_name.lower().replace(" ", "_")}.pdf', format='pdf', dpi=300)
    plt.close()