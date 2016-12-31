
# coding: utf-8

# In[2]:

import numpy as np


# - Data types
# - Array types
# - Type conversions
# - Creating Arrays
# - Indexing
# - Fancy Indexing
# - Slicing
# - Manipualting shapes
# 

# # The Numpy array object

# NumPy has a multidimensional array object called * ndarray *
# - The Actual data
# - Some metadata describing the data

# In[3]:

a = np.arange(5)
a.dtype


# In[4]:

a


# In[5]:

a.shape


# ### Creating a multidimensional array

# In[8]:

m = np.array([np.arange(2), np.arange(2)])
m


# In[9]:

m.shape


# ### Selecting array elements

# In[10]:

b = np.array([[1,2], [3,4]])
b


# In[11]:

b[0,0]


# In[12]:

b[0,1]


# In[13]:

b[1,0]


# ### NumPy numerical types

# In[20]:

print np.float32(42)
print np.int8(42.0)
print np.bool(42)
print np.bool(0)
print np.float(True)


# ### Data type objects

# In[22]:

# tells you the size of data in bytes
a.dtype.itemsize


# ### One-dimensional slicing and indexing

# In[27]:

a = np.arange(9)
print a
a[3:7]


# In[28]:

a[:7:2]


# In[29]:

a[::-1]


# ### Manipulating array shapes

# In[31]:

b = np.array([[[ 0, 1, 2, 3],
                [ 4, 5, 6, 7],
                [ 8, 9, 10, 11]],
                [[12, 13, 14, 15],
                [16, 17, 18, 19],
                [20, 21, 22, 23]]])
b


# In[32]:

b.ravel()


# In[33]:

b.flatten()


# In[34]:

b.shape


# In[36]:

b.shape = (6,4)
b


# In[41]:

b.transpose()


# In[43]:

b.resize((2,12))
b


# ### Stacking arrays

# In[46]:

a = np.arange(9).reshape(3,3)
a


# In[48]:

b = 2 * a
b


# In[50]:

# Horizontal stacking
np.hstack((a,b))


# In[52]:

np.concatenate((a,b), axis=1)


# In[53]:

# Vertical Stacking
np.vstack((a, b))


# In[54]:

np.concatenate((a,b), axis=0)


# In[55]:

# Depth Stacking
np.dstack((a,b))


# In[57]:

# Column stacking
oned = np.arange(2)
oned


# In[58]:

twice_oned = 2 * oned
twice_oned


# In[59]:

np.column_stack((oned, twice_oned))


# In[60]:

np.column_stack((a,b))


# In[64]:

np.column_stack((a,b)) == np.hstack((a,b))


# In[65]:

# Row Stacking
np.row_stack((oned, twice_oned))


# In[66]:

np.row_stack((a,b))


# In[67]:

np.row_stack == np.vstack((a,b))


# ### Splitting arrays

# In[68]:

a


# In[70]:

# Horizontal Splitting
np.hsplit(a, 3)


# In[71]:

np.split(a, 3, axis=1)


# In[72]:

# Vertical Splitting
np.vsplit(a, 3)


# In[74]:

# Depth-Wise Splitting
c = np.arange(27).reshape(3,3,3)
c


# In[76]:

np.dsplit(c, 3)


# ###  Array attributes

# In[78]:

b


# In[80]:

# gives the number of array dimensions
b.ndim


# In[81]:

# displays the number of elements
b.size


# In[82]:

# gives the number of bytes for each element in an array
b.itemsize


# In[83]:

# gives the total number of bytes an array requires
b.nbytes


# In[86]:

# has same effect as transpose() function
b.resize(1,9)
b


# In[87]:

b.T


# In[88]:

# Complex numbers
b = np.array([1.j + 1, 2.j + 3])
b


# In[89]:

b.real


# In[90]:

b.imag


# In[91]:

b.dtype


# In[93]:

# returns a numpy.flatiter object
b = np.arange(4).reshape(2, 2)
b


# In[95]:

f = b.flat
f


# In[96]:

for item in f:
    print item


# In[97]:

b.flat[2]


# In[98]:

b.flat[[1,3]]


# In[99]:

# The flat attribute is settable
b.flat = 7
b


# In[100]:

b.flat[[1,3]] = 1
b


# ### Converting arrays

# In[103]:

b = np.array([1. + 2.j, 3. + 4.j])
b


# In[104]:

# convert an array to a list
b.tolist()


# In[105]:

# convert an array to an array of the specified type 
b


# In[106]:

b.astype(int)


# In[107]:

b.astype('complex')


# ### Creating views and copies

# In[125]:

import scipy.misc
import matplotlib.pyplot as plt


# In[126]:

face = scipy.misc.face()
acopy = face.copy()
aview = face.view()


# In[131]:

plt.subplot(221)
plt.imshow(face)
plt.subplot(222)
plt.imshow(acopy)
plt.subplot(223)
plt.imshow(aview)

aview.flat = 1
plt.subplot(224)
plt.imshow(aview)
plt.show()
None


# In[132]:

ascent = scipy.misc.ascent()
acopy = ascent.copy()
aview = ascent.view()


# In[134]:

plt.subplot(221)
plt.imshow(ascent)
plt.subplot(222)
plt.imshow(acopy)
plt.subplot(223)
plt.imshow(aview)

aview.flat = 1
plt.subplot(224)
plt.imshow(aview)
plt.show()
None


# ### Fancy indexing

# In[145]:

ascent = scipy.misc.ascent()
xmax = ascent.shape[0]
ymax = ascent.shape[1]
ascent


# In[144]:

ascent[range(xmax), range(ymax)] = 0
print ascent
ascent[range(xmax-1, -1, -1), range(ymax)] = 0
plt.imshow(ascent)
plt.show()


# We defined separate ranges for the x and y values. These ranges were used to index the Ascent array. Fancy indexing is performed based on an internal NumPy iterator object. This can be achieved by performing the following three steps:
# 1. The iterator object is created.
# 2. The iterator object gets bound to the array.
# 3. Array elements are accessed via the iterator.

# ### Indexing with a list of locations

# In[146]:

ix_([0,1],[2,3])


# To index the array with a list of locations, perform the following steps:
# 1. Shuffle the array indices.

# In[150]:

ascent = scipy.misc.ascent()
xmax = ascent.shape[0]
ymax = ascent.shape[1]


# In[152]:

def shuffle_indices(size):
    arr = np.arange(size)
    np.random.shuffle(arr)
    
    return arr


# In[153]:

xindices = shuffle_indices(xmax)
np.testing.assert_equal(len(xindices), xmax)
yindices = shuffle_indices(ymax)
np.testing.assert_equal(len(yindices), ymax)

plt.imshow(ascent[np.ix_(xindices, yindices)])
plt.show()


# ### Indexing arrays with Booleans

# Boolean indexing is indexing based on a Boolean array and falls in the category of
# fancy indexing. Since Boolean indexing is a form of fancy indexing, the way it works
# is basically the same. This means that indexing happens with the help of a special
# iterator object.

# In[154]:

ascent = scipy.misc.ascent()


# In[155]:

"""create an image with dots on the diagonal and
   select modulo four points on the diagonal of the image"""
def get_indices(size):
    arr = np.arange(size)
    
    return arr % 4 == 0


# In[157]:

"""just apply this selection and plot the points"""

ascent1 = ascent.copy()
xindices = get_indices(ascent.shape[0])
yindices = get_indices(ascent.shape[1])
ascent1[xindices, yindices] = 0
plt.subplot(211)
plt.imshow(ascent1)


# In[159]:

"""Select array values between a quarter and three-quarters of the maximum
value, and set them to 0"""

ascent2 = ascent.copy()
ascent2[(ascent > ascent.max() / 4 ) & ( ascent < 3 * ascent.max()/4)] = 0
plt.subplot(212)
plt.imshow(ascent2)
plt.show()


# ###  Stride tricks for Sudoku

# In[166]:

sudoku = np.array([
        [8, 2, 7, 1, 6, 5, 9, 4, 3],
        [9, 6, 5, 3, 2, 7, 1, 4, 8],
        [3, 4, 1, 6, 8, 9, 7, 5, 2],
        [5, 9, 3, 4, 6, 8, 2, 7, 1],
        [4, 7, 2, 5, 1, 3, 6, 8, 9],
        [6, 1, 8, 9, 7, 2, 4, 3, 5],
        [7, 8, 6, 2, 3, 5, 9, 1, 4],
        [1, 5, 4, 7, 9, 6, 8, 2, 3],
        [2, 3, 9, 8, 4, 1, 5, 6, 7]
        
    ])
shape = (3,3,3,3)


# In[167]:

# The itemsize field of ndarray gives us the number
# of bytes in an array. itemsize calculates the strides
strides = sudoku.itemsize * np.array([27, 3, 9, 1])
strides


# In[168]:

# Now we can split the puzzle into squares with the as_strided() function of
# the np.lib.stride_tricks module

squares = np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides)
print squares


# ### Broadcasting arrays

# In[169]:

import scipy.io.wavfile
import urllib2


# In[181]:

response = urllib2.urlopen('http://www.thesoundarchive.com/austinpowers/smashingbaby.wav')
print response.info()


# In[182]:

WAV_FILE = 'smashingbaby.wav'
filehandle = open(WAV_FILE, 'w')
filehandle.write(response.read())
filehandle.close()
sample_rate, data = scipy.io.wavfile.read(WAV_FILE)
print "Data type", data.dtype, "Shape", data.shape


# In[183]:

plt.subplot(211)
plt.title('Original')
plt.plot(data)


# In[184]:

newdata = data * 0.2
newdata = newdata.astype(np.uint8)
print "Data type", newdata.dtype, "Shape", newdata.shape
scipy.io.wavfile.write("quiet.wav", sample_rate, newdata)
plt.subplot(212)
plt.title("Quiet")
plt.plot(newdata)
plt.show()


# In[ ]:



