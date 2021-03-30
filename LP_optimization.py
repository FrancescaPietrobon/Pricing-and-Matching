from scipy.optimize import linprog

def LP(price, p1, p2, p3,
       pr_c1_p0, pr_c2_p0, pr_c3_p0, pr_c4_p0,
       pr_c1_p1, pr_c2_p1, pr_c3_p1, pr_c4_p1,
       pr_c1_p2, pr_c2_p2, pr_c3_p2,pr_c4_p2,
       pr_c1_p3, pr_c2_p3, pr_c3_p3, pr_c4_p3,
       max_p0, max_p1, max_p2, max_p3,
       max_n1, max_n2, max_n3, max_n4):


    c = [price*pr_c1_p0, price*pr_c2_p0, price*pr_c3_p0, price*pr_c4_p0,
         price*pr_c1_p1*p1, price*pr_c2_p1*p1, price*pr_c3_p1*p1, price*pr_c4_p1*p1,
         price*pr_c1_p2*p2, price*pr_c2_p2*p2, price*pr_c3_p2*p2, price*pr_c4_p2*p2,
         price*pr_c1_p3*p3, price*pr_c2_p3*p3, price*pr_c3_p3*p3, price*pr_c4_p3*p3]

    A = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
         ]

    b = [max_p0, max_p1, max_p2, max_p3, max_n1, max_n2, max_n3, max_n4]

    x0_bounds = (0, None)
    x1_bounds = (0, None)
    x2_bounds = (0, None)
    x3_bounds = (0, None)
    x4_bounds = (0, None)
    x5_bounds = (0, None)
    x6_bounds = (0, None)
    x7_bounds = (0, None)
    x8_bounds = (0, None)
    x9_bounds = (0, None)
    x10_bounds = (0, None)
    x11_bounds = (0, None)
    x12_bounds = (0, None)
    x13_bounds = (0, None)
    x14_bounds = (0, None)
    x15_bounds = (0, None)

    res = linprog(c, A, b, [x0_bounds, x1_bounds, x2_bounds, x3_bounds,
                                             x4_bounds, x5_bounds, x6_bounds, x7_bounds,
                                             x8_bounds, x9_bounds, x10_bounds,x11_bounds,
                                             x12_bounds, x13_bounds, x14_bounds, x15_bounds])
    return res