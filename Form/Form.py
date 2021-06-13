import pandas as pd
import numpy as np


class Form:
    def __init__(self):
        # Importing the csv containing the form results
        csv = pd.read_csv('./Form/DIA_Form.csv')
        csv.columns = ['Date', 'Age', 'Sex', 'AW', 'WS0', 'WS10', 'WS20', 'WS50']

        # Classes of customers
        c1 = csv[(csv.Sex == 'Femmina') & (csv.Age <= 35)]               # C1: Females under 35
        c2 = csv[(csv.Sex == 'Maschio') & (csv.Age <= 35)]               # C2: Males under 35
        c3 = csv[(csv.Sex == 'Femmina') & (csv.Age > 35)]                # C3: Females over 35
        c4 = csv[(csv.Sex == 'Maschio') & (csv.Age > 35)]                # C4: Males over 35

        # Total number of customers per class
        n1 = len(c1.Date)
        n2 = len(c2.Date)
        n3 = len(c3.Date)
        n4 = len(c4.Date)
        self.n = np.array([n1, n2, n3, n4])

        # Number of people per class that buy (at least) item 1
        nc1_i1 = c1.Date[c1.AW == 'Sì'].count()
        nc2_i1 = c2.Date[c2.AW == 'Sì'].count()
        nc3_i1 = c3.Date[c3.AW == 'Sì'].count()
        nc4_i1 = c4.Date[c4.AW == 'Sì'].count()

        # Probability (per class) that a customer buys (at least)
        c1_i1 = nc1_i1 / n1
        c2_i1 = nc2_i1 / n2
        c3_i1 = nc3_i1 / n3
        c4_i1 = nc4_i1 / n4
        self.i1_param = np.array([c1_i1, c2_i1, c3_i1, c4_i1])

        # Number of customers per class that buy item 2 after buying item 1 and getting promo P0
        nc1_i2_p0 = c1.Date[c1.WS0 == 'Sì'].count()
        nc2_i2_p0 = c2.Date[c2.WS0 == 'Sì'].count()
        nc3_i2_p0 = c3.Date[c3.WS0 == 'Sì'].count()
        nc4_i2_p0 = c4.Date[c4.WS0 == 'Sì'].count()

        # Probability (per class) that a customer buys item 2 after buying item 1 and getting promo P0
        c1_i21_p0_param = nc1_i2_p0 / n1
        c2_i21_p0_param = nc2_i2_p0 / n2
        c3_i21_p0_param = nc3_i2_p0 / n3
        c4_i21_p0_param = nc4_i2_p0 / n4

        # Number of customers per class that buy item 2 after buying item 1 and getting promo P1
        nc1_i2_p1 = c1.Date[c1.WS10 == 'Sì'].count()
        nc2_i2_p1 = c2.Date[c2.WS10 == 'Sì'].count()
        nc3_i2_p1 = c3.Date[c3.WS10 == 'Sì'].count()
        nc4_i2_p1 = c4.Date[c4.WS10 == 'Sì'].count()

        # Probability (per class) that a customer buys item 2 after buying item 1 and getting promo P1
        c1_i21_p1_param = (nc1_i2_p1 + nc1_i2_p0) / n1
        c2_i21_p1_param = (nc2_i2_p1 + nc2_i2_p0) / n2
        c3_i21_p1_param = (nc3_i2_p1 + nc3_i2_p0) / n3
        c4_i21_p1_param = (nc4_i2_p1 + nc4_i2_p0) / n4

        # Number of customers per class that buy item 2 after buying item 1 and getting promo P2
        nc1_i2_p2 = c1.Date[c1.WS20 == 'Sì'].count()
        nc2_i2_p2 = c2.Date[c2.WS20 == 'Sì'].count()
        nc3_i2_p2 = c3.Date[c3.WS20 == 'Sì'].count()
        nc4_i2_p2 = c4.Date[c4.WS20 == 'Sì'].count()

        # Probability (per class) that a customer buys item 2 after buying item 1 and getting promo P2
        c1_i21_p2_param = (nc1_i2_p2 + nc1_i2_p1 + nc1_i2_p0) / n1
        c2_i21_p2_param = (nc2_i2_p2 + nc2_i2_p1 + nc2_i2_p0) / n2
        c3_i21_p2_param = (nc3_i2_p2 + nc3_i2_p1 + nc3_i2_p0) / n3
        c4_i21_p2_param = (nc4_i2_p2 + nc4_i2_p1 + nc4_i2_p0) / n4

        # Number of customers per class that buy item 2 after buying item 1 and getting promo P3
        nc1_i2_p3 = c1.Date[c1.WS50 == 'Sì'].count()
        nc2_i2_p3 = c2.Date[c2.WS50 == 'Sì'].count()
        nc3_i2_p3 = c3.Date[c3.WS50 == 'Sì'].count()
        nc4_i2_p3 = c4.Date[c4.WS50 == 'Sì'].count()

        # Probability (per class) that a customer buys item 2 after buying item 1 and getting promo P3
        c1_i21_p3_param = (nc1_i2_p3 + nc1_i2_p2 + nc1_i2_p1 + nc1_i2_p0) / n1
        c2_i21_p3_param = (nc2_i2_p3 + nc2_i2_p2 + nc2_i2_p1 + nc2_i2_p0) / n2
        c3_i21_p3_param = (nc3_i2_p3 + nc3_i2_p2 + nc3_i2_p1 + nc3_i2_p0) / n3
        c4_i21_p3_param = (nc4_i2_p3 + nc4_i2_p2 + nc4_i2_p1 + nc4_i2_p0) / n4

        self.i21_param = np.array([[c1_i21_p0_param, c2_i21_p0_param, c3_i21_p0_param, c4_i21_p0_param],
                                   [c1_i21_p1_param, c2_i21_p1_param, c3_i21_p1_param, c4_i21_p1_param],
                                   [c1_i21_p2_param, c2_i21_p2_param, c3_i21_p2_param, c4_i21_p2_param],
                                   [c1_i21_p3_param, c2_i21_p3_param, c3_i21_p3_param, c4_i21_p3_param]])
