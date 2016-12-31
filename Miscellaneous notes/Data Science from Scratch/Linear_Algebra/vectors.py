height_weight_age = [70,  # inches,
                     170,  # pounds,
                     40]  # years

grades = [95,  # exam1
          80,  # exam2
          75,  # exam3
          62]  # exam4

# use zip the vectors and list comprehension for arithmetic operations


def vector_add(v, w):
    """adds corresponding elements"""
    return [v_i + w_i for v_i , w_i in zip(v, w)]


def vector_subtract(v, w):
    """subtracts corresponding elements"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors):
    """sums all corresponding elements"""
    result = vectors[0]             # start with the first vector
    for vector in vectors[1:]:      # then loop over the others
        result = vector_add(result, vector) # and add them to the result


def scalar_multiply(c, v):
    """c is a number, v is a vector"""
    return [c * v_i for v_i in v]


def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot(v, w):
    return sum(v_i * w_i
               for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    return dot(v, v)


def magnitude(v):
    return math.sqrt(sum_of_squares(v))


def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))


def distance(v, w):
    return math.sqrt(squared_distance(v, w))


