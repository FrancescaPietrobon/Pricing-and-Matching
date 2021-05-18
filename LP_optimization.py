from scipy.optimize import linprog


def LP(price_item2, discounts, prob_buy_item21, daily_promos, daily_customers):

    # Objective Function
    c = [price_item2 * prob_buy_item21[0][0], price_item2 * prob_buy_item21[0][1], price_item2 * prob_buy_item21[0][2], price_item2 * prob_buy_item21[0][3],
         price_item2 * prob_buy_item21[1][0] * (1-discounts[1]), price_item2 * prob_buy_item21[1][1] * (1-discounts[1]), price_item2 * prob_buy_item21[1][2] * (1-discounts[1]), price_item2 * prob_buy_item21[1][3] * (1-discounts[1]),
         price_item2 * prob_buy_item21[2][0] * (1-discounts[2]), price_item2 * prob_buy_item21[2][1] * (1-discounts[2]), price_item2 * prob_buy_item21[2][2] * (1-discounts[2]), price_item2 * prob_buy_item21[2][3] * (1-discounts[2]),
         price_item2 * prob_buy_item21[3][0] * (1-discounts[3]), price_item2 * prob_buy_item21[3][1] * (1-discounts[3]), price_item2 * prob_buy_item21[3][2] * (1-discounts[3]), price_item2 * prob_buy_item21[3][3] * (1-discounts[3])]

    # First model: maximum number of promo codes is the number of promo codes for P1, P2 and P3
    A_inequality = [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]

    A_equality = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]

    b_inequality = [daily_customers[0], daily_customers[1], daily_customers[2], daily_customers[3]]

    b_equality = [daily_promos[0], daily_promos[1], daily_promos[2], daily_promos[3]]

    # Variables
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

    # Optimization
    res = linprog(c=c, A_eq=A_equality, b_eq=b_equality, A_ub=A_inequality, b_ub=b_inequality,
                  bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds,
                          x4_bounds, x5_bounds, x6_bounds, x7_bounds,
                          x8_bounds, x9_bounds, x10_bounds, x11_bounds,
                          x12_bounds, x13_bounds, x14_bounds, x15_bounds])

    result = res.x.astype(int)
    result = result.reshape((4, 4))

    return res.fun, result
