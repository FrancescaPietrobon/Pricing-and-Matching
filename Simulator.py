#class Simulator:
    # Number of customers per class = potentially different Gaussian distributions
    # Class 1: 76 customers, sigma 12
    # Class 2: 133 customers, sigma 14
    # Class 3: 107 customers, sigma 16
    # Class 4: 93 customers, sigma 17

    # Probability that a customer of a class buys the first object = Binomial
    # Class 1: N = 76, p = 0.632
    # Class 2: N = 133, p = 0.114
    # Class 3: N = 107, p = 0.333
    # Class 4: N = 93, p = 0.713

    # Probability that a customer of a class buys the second object = Binomial
    # Class 1: N = 76, p = 0.276
    # Class 2: N = 133, p = 0.421
    # Class 3: N = 107, p = 0.358
    # Class 4: N = 93, p = 0.452

    # Probability that a customer of a class buys the first object given the second + P0 = Binomial
    # Class 1: N = 76, p = 0.862
    # Class 2: N = 133, p = 0.920
    # Class 3: N = 107, p = 0.267
    # Class 4: N = 93, p = 0.148

    # Probability that a customer of a class buys the second object given the first + P0 = Binomial
    # Class 1: N = 76, p = 0.512
    # Class 2: N = 133, p = 0.830
    # Class 3: N = 107, p = 0.122
    # Class 4: N = 93, p = 0.010

    # Probability that a customer of a class buys the first object given the second + P1 = Binomial
    # Class 1: N = 76, p = 0.840
    # Class 2: N = 133, p = 0.921
    # Class 3: N = 107, p = 0.270
    # Class 4: N = 93, p = 0.169

    # Probability that a customer of a class buys the second object given the first + P1 = Binomial
    # Class 1: N = 76, p = 0.845
    # Class 2: N = 133, p = 0.831
    # Class 3: N = 107, p = 0.145
    # Class 4: N = 93, p = 0.201

    # Probability that a customer of a class buys the first object given the second + P2 = Binomial
    # Class 1: N = 76, p = 0.862
    # Class 2: N = 133, p = 0.921
    # Class 3: N = 107, p = 0.481
    # Class 4: N = 93, p = 0.253

    # Probability that a customer of a class buys the second object given the first + P2 = Binomial
    # Class 1: N = 76, p = 0.872
    # Class 2: N = 133, p = 0.872
    # Class 3: N = 107, p = 0.367
    # Class 4: N = 93, p = 0.364

    # Probability that a customer of a class buys the first object given the second + P3 = Binomial
    # Class 1: N = 76, p = 0.899
    # Class 2: N = 133, p = 0.750
    # Class 3: N = 107, p = 0.678
    # Class 4: N = 93, p = 0.759

    # Probability that a customer of a class buys the second object given the first + P3 = Binomial
    # Class 1: N = 76, p = 0.910
    # Class 2: N = 133, p = 0.700
    # Class 3: N = 107, p = 0.662
    # Class 4: N = 93, p = 0.546