import math
from matplotlib import pyplot as plt


# density function of the uniform distribution
def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0


# cumulative distribution function
def uniform_cdf(x):
    "returns the probability that a uniform random variable is <= x"
    if x < 0:
        return 0  # uniform random is never less than 0
    elif x < 1:
        return x  # e.g. P(x <= 0.4) = 0.4
    else:
        return 1  # uniform random is always less than 1


def normal_pdf(x, mu=0, sigma=1.0):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma)

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu = 0, sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu = 0, sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label='mu = 0, sigma=0.5')
plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.', label='mu = -1, sigma=1')

plt.legend()
plt.title("Various Normal pdfs")
plt.show()


def normal_cdf(x, mu=0, sigma=1.0):
    return (1 + math.erf((x-mu) / math.sqrt(2)/ sigma )) / 2

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-', label='mu = 0, sigma=1')
plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label='mu = 0, sigma=2')
plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':', label='mu = 0, sigma=0.5')
plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.', label='mu = -1, sigma=1')

plt.legend(loc=4)
plt.title("Various Normal cdfs")
plt.show()