import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

print(data[:10])
print(data.shape)

# Preprocessing and cleaning the data
# split the vectors in two vectors

x = data[:, 0]   # contains the hours
y = data[:, 1]   # contains the Web Hits in that particular hour

# check how many hours contain invalid data
sp.sum(sp.isnan(y))

# choose  those elements from x and y where y contains valid numbers
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

# plot Data in a scatter plot
# plot the (x, y) points with dots of size 10
plt.scatter(x, y, s=10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([ w * 7 * 24 for w in range(10)],
           ['week %i' % w for w in range(10)])
plt.autoscale(tight=True)


# draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-', color='0.75')


# choosing the right model and learning algorithm
def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)

# linear Model, polynomial of degree 1
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print ("Model parameters: %s" % fp1)
print (residuals)
f1 = sp.poly1d(fp1)
print ("error for linear model", error(f1, x, y))

# Quadratic model
f2p = sp.polyfit(x, y, 2)
f2 = sp.poly1d(f2p)
print("error for quadratic model", error(f2, x, y))

# Cubic model
f3p = sp.polyfit(x, y, 3)
f3 = sp.poly1d(f3p)
print("error for cubic model", error(f3, x, y))


# polynomial with degree 10
f10p = sp.polyfit(x, y, 10)
f10 = sp.poly1d(f10p)
print("error for model of degree d = 10", error(f10, x, y))


# polynomial with degree 50
f50p = sp.polyfit(x, y, 50)
f50 = sp.poly1d(f50p)
print("error for model of degree d = 50", error(f50, x, y))


# there is an inflection point between weeks 3 and 4
inflection = 3.5 * 7 * 24
xa = x[:inflection]  # calculate the inflection point in hours
ya = y[:inflection]
xb = x[inflection:]  # data after
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)

print ("Error inflection = %f" % (fa_error + fb_error))

fx = sp.linspace(0, x[-1], 1000)  # generate X-values for plotting
plt.plot(fx, f1(fx), linewidth=4)
plt.plot(fx, f2(fx), linewidth=4, color='red')
plt.plot(fx, f3(fx), linewidth=4, color='green')
plt.plot(fx, f10(fx), linewidth=4, linestyle='--', color='yellow')
plt.plot(fx, f50(fx), linewidth=4, linestyle=':', color='red')

plt.legend(["d=%i" % f1.order, "d=%i" % f2.order, "d=%i" % f3.order, "d=%i" % f10.order, "d=%i" % f50.order],
           loc="upper left")
plt.show()


