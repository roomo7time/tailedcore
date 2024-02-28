import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Make sure to import seaborn



def plot_bars(
    data_lists,
    list_names,
    x_ticks,
    y_label,
    filename,
    list_colors=None,
    alpha=0.75,
    width=0.4,
    x_tick_size=16,
    y_label_size=16,
    show=False,
    legend=True,
):
    if list_colors is None:
        # Create a colormap and generate colors from it
        cmap = plt.cm.get_cmap("hsv", len(data_lists))
        list_colors = [cmap(i) for i in range(cmap.N)]  # Generate colors

    n = len(data_lists[0])  # Assuming all lists are of the same length
    r = np.arange(n)  # Initial positions of the bars
    bar_width = width / len(data_lists)  # Adjust the width based on number of lists

    for i, (data_list, name) in enumerate(zip(data_lists, list_names)):
        # Calculate position for each bar
        pos = [x + (bar_width * i) for x in r]

        # Creating the bar plot for each list
        plt.bar(
            pos,
            data_list,
            color=list_colors[i],
            width=bar_width,
            label=name,
            alpha=alpha,
        )

    # Adding labels
    plt.ylabel(y_label, size=y_label_size)
    plt.xticks(
        [r + width / 2 - bar_width / 2 for r in range(n)], x_ticks, size=x_tick_size
    )

    # Adding a legend
    if legend:
        plt.legend()

    # Save the figure before showing it
    plt.savefig(filename, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


# def plot_and_save_correlation_graph(
#     scores1,
#     scores2,
#     score1_name='Score 1',
#     score2_name='Score 2',
#     alpha=0.0025,
#     color='#FEB24c',    # '#9CCEA7' '#FEB24c' '#FED976'
#     polyfit=False,
#     polyfit_color='#0D4A70',
#     file_name="./correlation_graph.png",
# ):
#     if len(scores1) != len(scores2):
#         raise ValueError("Both score arrays must have the same length.")

#     plt.scatter(scores1, scores2, alpha=alpha, color=color)
#     plt.xlabel(score1_name)
#     plt.ylabel(score2_name)

#     if polyfit:
#         # Randomly sample approximately 1% of the data points
#         sample_size = max(1, len(scores1) // 100)  # Ensure at least 1 sample is taken
#         random_indices = np.random.choice(len(scores1), size=sample_size, replace=False)
#         sampled_scores1 = np.array(scores1)[random_indices]
#         sampled_scores2 = np.array(scores2)[random_indices]

#         m, b = np.polyfit(sampled_scores1, sampled_scores2, 1)
#         plt.plot(scores1, m * np.array(scores1) + b, color=polyfit_color)

#     plt.xlim(0.85, 2.25)
#     plt.tight_layout()  # Adjust the layout to make sure everything fits without overlapping
#     plt.savefig(file_name)
#     plt.close()  # Close the figure to free memory

def plot_and_save_correlation_graph(
    scores1,
    scores2,
    labels,
    label_colors=None,
    label_names=None,
    score1_name='Score 1',
    score2_name='Score 2',
    alpha=0.0025,
    polyfit=False,
    polyfit_color='#0D4A70',
    file_name="./correlation_graph.png",
):
    if len(scores1) != len(scores2) or len(scores1) != len(labels):
        raise ValueError("Scores and labels arrays must have the same length.")

    unique_labels = np.unique(labels)
    
    if label_colors is None:
        num_colors = len(unique_labels)
        palette = sns.color_palette("Spectral", num_colors)  # You can change "viridis" to any other Seaborn palette
        label_colors = [palette[i] for i, label in enumerate(unique_labels)]
    
    label_colors = [label_colors[0], label_colors[3], label_colors[2], label_colors[1]]
    
    if len(unique_labels) != len(label_colors):
        raise ValueError("Number of unique labels and label_colors must match.")

    # Plot each group with its corresponding color
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)
        plt.scatter(np.array(scores1)[idx], np.array(scores2)[idx], alpha=alpha, color=label_colors[i], label=label_names[i])

    plt.xlabel(score1_name)
    plt.ylabel(score2_name)
    leg = plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    if polyfit:
        # Randomly sample approximately 1% of the data points for the polyfit calculation
        sample_size = max(1, len(scores1) // 100)  # Ensure at least 1 sample is taken
        random_indices = np.random.choice(len(scores1), size=sample_size, replace=False)
        sampled_scores1 = np.array(scores1)[random_indices]
        sampled_scores2 = np.array(scores2)[random_indices]

        m, b = np.polyfit(sampled_scores1, sampled_scores2, 1)
        plt.plot(scores1, m * np.array(scores1) + b, color=polyfit_color)

    plt.xlim(0.85, 2.25)
    plt.tight_layout()  # Adjust the layout
    plt.savefig(file_name)
    plt.close()


if __name__ == "__main__":
    # Example test code
    list1 = [10, 20, 30, 40]
    list2 = [15, 25, 35, 45]
    list1_name = "Group A"
    list2_name = "Group B"
    x_ticks = ["Q1", "Q2", "Q3", "Q4"]
    y_label = "Revenue"
    filename = "./my_custom_bar_plot.png"
