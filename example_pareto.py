import numpy as np
from scipy.stats import pareto
import matplotlib.pyplot as plt


def get_discrete_pareto_pmf(alpha, sampe_space_size, epsilon=0.01):
    assert epsilon > 0
    x = np.linspace(
        pareto.ppf(epsilon, alpha), pareto.ppf(1 - epsilon, alpha), sampe_space_size + 1
    )
    cdf = pareto.cdf(x, alpha)
    pmf = (cdf - np.concatenate([[0], cdf])[:-1])[1:]

    return pmf


if __name__ == "__main__":

    integers = [5, 12, 3, 8, 15, 6, 10, 20, 7, 11, 9, 14, 2, 18, 4]

    alpha = 6.0
    pmf = get_discrete_pareto_pmf()

    fig, ax = plt.subplots(1, 1)

    plt.show()
