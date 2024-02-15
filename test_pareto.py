import numpy as np
from scipy.stats import pareto
import matplotlib.pyplot as plt

# # Pareto distribution parameters
# a = 100 # shape parameter
# b = 6
# size = 1400 # number of samples

# # Generating samples
# samples = np.random.pareto(a, size)

# # Creating the histogram
# bin_count = 15  # Number of bins
# counts, bin_edges = np.histogram(samples, bins=bin_count)

# # Displaying the counts
# print("Counts per bin:", counts)

# # Optional: Plotting the histogram for visualization
# plt.hist(samples, bins=bin_edges, edgecolor='black')
# plt.title('Pareto Distribution Histogram')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()
# plt.close()
# plt.clf()


# fig, ax = plt.subplots(1, 1)

# b = 6.0
# mean, var, skew, kurt = pareto.stats(b, moments='mvsk')

# x = np.linspace(pareto.ppf(0.01, b),
#                 pareto.ppf(0.99, b), 100)
# ax.plot(x, pareto.pdf(x, b),
#        'r-', lw=5, alpha=0.6, label='pareto pdf')

# plt.show()


integers = [5, 12, 3, 8, 15, 6, 10, 20, 7, 11, 9, 14, 2, 18, 4]

# Parameters for Pareto distribution
a = 6  # Shape parameter
x_m = 1  # Scale parameter

# Generate evenly spaced points
points = np.linspace(1, len(integers), len(integers))

# Calculate Pareto PDF values
pdf_values = pareto.pdf(points, a)

fig, ax = plt.subplots(1, 1)
ax.plot(points, pdf_values, "r-", lw=5, alpha=0.6, label="pareto pdf")

plt.show()
