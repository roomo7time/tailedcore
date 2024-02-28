import matplotlib.pyplot as plt
import numpy as np


def plot_bars(
    list1,
    list2,
    list1_name,
    list2_name,
    x_ticks,
    y_label,
    filename,
    list1_color="#FF7F0E",  
    # list2_color="#1F77B4",
    # list1_color="#FA7268",  
    list2_color="#17A2B8",
    alpha=0.75,
    width=0.4,
    x_tick_size=16,
    y_label_size=16,
    show=False,
):
    # Number of bars pairs
    n = len(list1)

    # Positions of the left bar-groups on the x-axis
    r1 = np.arange(n)

    # Positions of the right bar-groups
    r2 = [x + 0.4 for x in r1]

    # Creating the bar plot
    plt.bar(r1, list1, color=list1_color, width=width, label=list1_name, alpha=alpha)
    plt.bar(r2, list2, color=list2_color, width=width, label=list2_name, alpha=alpha)

    # Adding labels
    plt.ylabel(y_label, size=y_label_size)
    plt.xticks([r + 0.2 for r in range(n)], x_ticks, size=x_tick_size)

    # Adding a legend
    plt.legend()

    # Save the figure before showing it
    plt.savefig(filename, bbox_inches="tight")

    if show:
        plt.show()

def plot_bars_three_lists(
    list1,
    list2,
    list3,
    list1_name,
    list2_name,
    list3_name,
    x_ticks,
    y_label,
    filename,
    list1_color="#6BCB77",
    list2_color="#17A2B8",
    list3_color="#FF7F0E",
    alpha=0.75,
    width=0.25,  # Adjusted width for three bars
    x_tick_size=16,
    y_label_size=16,
    show=False,
):
    # Number of bars groups
    n = len(list1)

    # Positions of the bar groups on the x-axis
    r1 = np.arange(n)
    r2 = [x + width for x in r1]
    r3 = [x + width * 2 for x in r1]

    # Creating the bar plot
    plt.bar(r1, list1, color=list1_color, width=width, label=list1_name, alpha=alpha)
    plt.bar(r2, list2, color=list2_color, width=width, label=list2_name, alpha=alpha)
    plt.bar(r3, list3, color=list3_color, width=width, label=list3_name, alpha=alpha)

    # Adding labels
    plt.ylabel(y_label, size=y_label_size)
    plt.xticks([r + width for r in range(n)], x_ticks, size=x_tick_size)

    # Adding a legend
    plt.legend()

    # Save the figure before showing it
    plt.savefig(filename, bbox_inches="tight")

    if show:
        plt.show()

if __name__ == "__main__":
    # Example test code
    list1 = [10, 20, 30, 40]
    list2 = [15, 25, 35, 45]
    list1_name = 'Group A'
    list2_name = 'Group B'
    x_ticks = ['Q1', 'Q2', 'Q3', 'Q4']
    y_label = 'Revenue'
    filename = './my_custom_bar_plot.png'

    # Call the function with the example data
    plot_bars(list1, list2, list1_name, list2_name, x_ticks,  y_label, filename, show=True)