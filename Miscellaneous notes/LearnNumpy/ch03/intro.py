import numpy as np
import sys

to_float = lambda x: float(x.strip() or np.nan)

min_temp, max_temp = np.loadtxt("temp.csv", delimiter=',',
                                usecols=(7, 8), unpack=True, converters={7: to_float, 8: to_float}) * .1

print "# Records ", len(min_temp), len(max_temp)
print "Minimum ", np.nanmin(min_temp)
print "Maximum ", np.nanmax(max_temp)