from scipy.optimize import linprog
import warnings
warnings.filterwarnings(action='ignore', module='scipy')


def matching_lp(margin_item2, discounts, conversion_rates_item2, daily_promos, daily_customers):

    # Objective Function
    c = [margin_item2 * conversion_rates_item2[0][0] * (1 - discounts[0]),          # P0 - Class1
         margin_item2 * conversion_rates_item2[0][1] * (1 - discounts[0]),          # P0 - Class2
         margin_item2 * conversion_rates_item2[0][2] * (1 - discounts[0]),          # P0 - Class3
         margin_item2 * conversion_rates_item2[0][3] * (1 - discounts[0]),          # P0 - Class4
         margin_item2 * conversion_rates_item2[1][0] * (1 - discounts[1]),          # P1 - Class1
         margin_item2 * conversion_rates_item2[1][1] * (1 - discounts[1]),          # P1 - Class2
         margin_item2 * conversion_rates_item2[1][2] * (1 - discounts[1]),          # P1 - Class3
         margin_item2 * conversion_rates_item2[1][3] * (1 - discounts[1]),          # P1 - Class4
         margin_item2 * conversion_rates_item2[2][0] * (1 - discounts[2]),          # P2 - Class1
         margin_item2 * conversion_rates_item2[2][1] * (1 - discounts[2]),          # P2 - Class2
         margin_item2 * conversion_rates_item2[2][2] * (1 - discounts[2]),          # P2 - Class3
         margin_item2 * conversion_rates_item2[2][3] * (1 - discounts[2]),          # P2 - Class4
         margin_item2 * conversion_rates_item2[3][0] * (1 - discounts[3]),          # P3 - Class1
         margin_item2 * conversion_rates_item2[3][1] * (1 - discounts[3]),          # P3 - Class2
         margin_item2 * conversion_rates_item2[3][2] * (1 - discounts[3]),          # P3 - Class3
         margin_item2 * conversion_rates_item2[3][3] * (1 - discounts[3])]          # P3 - Class4

    # Inequality constraints: the sum of promo codes given to a class does not exceed the customers of that class
    a_inequality = [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],       # Class1
                    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],       # Class2
                    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],       # Class3
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]]       # Class4

    b_inequality = [daily_customers[0], daily_customers[1], daily_customers[2], daily_customers[3]]

    # Equality constraints: the sum of promo codes given to the classes is equal to the available number of promo codes
    a_equality = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],         # P0
                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],         # P1
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],         # P2
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]         # P3

    b_equality = [daily_promos[0], daily_promos[1], daily_promos[2], daily_promos[3]]

    # Variables: each one represents the number of a specific promo associated to a specific customer class
    bounds = [(0, None)] * 16

    # Optimization
    options = dict(cholesky=False, sym_pos=False, lstsq=True, presolve=True)
    result = linprog(c, a_inequality, b_inequality, a_equality, b_equality, bounds, options=options)

    # Reshaping and casting the variables values found
    matching_matrix = result.x.reshape((4, 4)).astype(int)

    return result.fun, matching_matrix
