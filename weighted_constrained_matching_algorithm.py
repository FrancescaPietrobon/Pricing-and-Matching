import numpy as np

def weighted_constrained_matching_algorithm(p, c, v, w):           # p = number of promos [p1, p2, p3] (as P0 are infinity), c = number of customers [c1, c2, c3, c4], v = constraints row promo / column classes (without P0 as it is infinity), w = conversion rates differentials matrix
    res = np.zeros((4, 4))

    while np.max(w) > 0:
        argmax = w.argmax()
        max_x = argmax // w.shape[1]
        max_y = argmax % w.shape[1]
        w[max_x, max_y] = 0
        value = v[max_x, max_y]
        # We assume that the total number of each promo code type is the sum on the corresponding row on v -> so input p should be useless
        if value <= c[max_y]:
            c[max_y] = c[max_y] - value
            res[max_x, max_y] = value
        else:
            res[max_x, max_y] = c[max_y]
            c[max_y] = 0

    # Then, we give P0 to all the remaining customers (first line)
    for j in range(0, 4):
        res[0, j] = c[j]

    return res                              # Returns a matrix containing the associations
