import pandas as pd
import numpy as np


class Form:
    def __init__(self):
        # Importing the csv containing the form results
        csv = pd.read_csv('DIA_Form.csv')
        csv.columns = ['Date', 'Age', 'Sex', 'AW', 'WS0', 'WS10', 'WS20', 'WS50']

        # Classes of customers
        c1 = csv[(csv.Sex == 'Femmina') & (csv.Age <= 35)]               # C1: Females under 35
        c2 = csv[(csv.Sex == 'Maschio') & (csv.Age <= 35)]               # C2: Males under 35
        c3 = csv[(csv.Sex == 'Femmina') & (csv.Age > 35)]                # C3: Females over 35
        c4 = csv[(csv.Sex == 'Maschio') & (csv.Age > 35)]                # C4: Males over 35

        # Total number of customers per class
        self.n1 = len(c1.Date)
        self.n2 = len(c2.Date)
        self.n3 = len(c3.Date)
        self.n4 = len(c4.Date)

        # Number of people per class that buy (at least) item 1
        nc1_i1 = c1.Date[c1.AW == 'Sì'].count()
        nc2_i1 = c2.Date[c2.AW == 'Sì'].count()
        nc3_i1 = c3.Date[c3.AW == 'Sì'].count()
        nc4_i1 = c4.Date[c4.AW == 'Sì'].count()

        # Probability (per class) that a customer buys (at least)
        self.c1_i1 = nc1_i1 / self.n1
        self.c2_i1 = nc2_i1 / self.n2
        self.c3_i1 = nc3_i1 / self.n3
        self.c4_i1 = nc4_i1 / self.n4

        # Number of customers per class that buy item 2 after buying item 1 and getting promo P0
        nc1_i2_p0 = c1.Date[c1.WS0 == 'Sì'].count()
        nc2_i2_p0 = c2.Date[c2.WS0 == 'Sì'].count()
        nc3_i2_p0 = c3.Date[c3.WS0 == 'Sì'].count()
        nc4_i2_p0 = c4.Date[c4.WS0 == 'Sì'].count()

        # Probability (per class) that a customer buys item 2 after buying item 1 and getting promo P0
        self.c1_i21_p0_param = nc1_i2_p0 / self.n1
        self.c2_i21_p0_param = nc2_i2_p0 / self.n2
        self.c3_i21_p0_param = nc3_i2_p0 / self.n3
        self.c4_i21_p0_param = nc4_i2_p0 / self.n4

        # Number of customers per class that buy item 2 after buying item 1 and getting promo P1
        nc1_i2_p1 = c1.Date[c1.WS10 == 'Sì'].count()
        nc2_i2_p1 = c2.Date[c2.WS10 == 'Sì'].count()
        nc3_i2_p1 = c3.Date[c3.WS10 == 'Sì'].count()
        nc4_i2_p1 = c4.Date[c4.WS10 == 'Sì'].count()

        # Probability (per class) that a customer buys item 2 after buying item 1 and getting promo P1
        self.c1_i21_p1_param = (nc1_i2_p1 + nc1_i2_p0) / self.n1
        self.c2_i21_p1_param = (nc2_i2_p1 + nc2_i2_p0) / self.n2
        self.c3_i21_p1_param = (nc3_i2_p1 + nc3_i2_p0) / self.n3
        self.c4_i21_p1_param = (nc4_i2_p1 + nc4_i2_p0) / self.n4

        # Number of customers per class that buy item 2 after buying item 1 and getting promo P2
        nc1_i2_p2 = c1.Date[c1.WS20 == 'Sì'].count()
        nc2_i2_p2 = c2.Date[c2.WS20 == 'Sì'].count()
        nc3_i2_p2 = c3.Date[c3.WS20 == 'Sì'].count()
        nc4_i2_p2 = c4.Date[c4.WS20 == 'Sì'].count()

        # Probability (per class) that a customer buys item 2 after buying item 1 and getting promo P2
        self.c1_i21_p2_param = (nc1_i2_p2 + nc1_i2_p1 + nc1_i2_p0) / self.n1
        self.c2_i21_p2_param = (nc2_i2_p2 + nc2_i2_p1 + nc2_i2_p0) / self.n2
        self.c3_i21_p2_param = (nc3_i2_p2 + nc3_i2_p1 + nc3_i2_p0) / self.n3
        self.c4_i21_p2_param = (nc4_i2_p2 + nc4_i2_p1 + nc4_i2_p0) / self.n4

        # Number of customers per class that buy item 2 after buying item 1 and getting promo P3
        nc1_i2_p3 = c1.Date[c1.WS50 == 'Sì'].count()
        nc2_i2_p3 = c2.Date[c2.WS50 == 'Sì'].count()
        nc3_i2_p3 = c3.Date[c3.WS50 == 'Sì'].count()
        nc4_i2_p3 = c4.Date[c4.WS50 == 'Sì'].count()

        # Probability (per class) that a customer buys item 2 after buying item 1 and getting promo P3
        self.c1_i21_p3_param = (nc1_i2_p3 + nc1_i2_p2 + nc1_i2_p1 + nc1_i2_p0) / self.n1
        self.c2_i21_p3_param = (nc2_i2_p3 + nc2_i2_p2 + nc2_i2_p1 + nc2_i2_p0) / self.n2
        self.c3_i21_p3_param = (nc3_i2_p3 + nc3_i2_p2 + nc3_i2_p1 + nc3_i2_p0) / self.n3
        self.c4_i21_p3_param = (nc4_i2_p3 + nc4_i2_p2 + nc4_i2_p1 + nc4_i2_p0) / self.n4

        # TODO:
        #  generate the number of customers that buy item 1 (in general, and of course with a promo since they get one after buying item 1)
        #  give promo: step 1 uniform; step 2 uniform first day, then use optimization data
        #  then, probability to buy also item 2 after receiving a promo (also P0)
        #  P(buy item 2 given P0) + P(not buy item 2 given P0) = 1 (for each class)
        #  P(buy item 2 given P1) + P(not buy item 2 given P1) = 1 (for each class)
        #  P(buy item 2 given P2) + P(not buy item 2 given P2) = 1 (for each class)
        #  P(buy item 2 given P3) + P(not buy item 2 given P3) = 1 (for each class)

    def get_n(self):
        return np.array([self.n1, self.n2, self.n3, self.n4])

    def get_i1_param(self):
        return np.array([self.c1_i1, self.c2_i1, self.c3_i1, self.c4_i1])

    def get_i21_param(self):
        return np.array([[self.c1_i21_p0_param, self.c2_i21_p0_param, self.c3_i21_p0_param, self.c4_i21_p0_param],
                         [self.c1_i21_p1_param, self.c2_i21_p1_param, self.c3_i21_p1_param, self.c4_i21_p1_param],
                         [self.c1_i21_p2_param, self.c2_i21_p2_param, self.c3_i21_p2_param, self.c4_i21_p2_param],
                         [self.c1_i21_p3_param, self.c2_i21_p3_param, self.c3_i21_p3_param, self.c4_i21_p3_param]])
