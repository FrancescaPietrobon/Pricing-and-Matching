
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\franc\OneDrive - Politecnico di Milano\POLIMI\Data Intelligence Applications\Prove\DIAForm2.csv")
data.columns = ['Date', 'Age', 'Sex', 'AW', 'WS0', 'WS10', 'WS20', 'WS50']


# Total number of people
N = len(data.Date)

#CLASSES
#C1: Female under 35
#C2: Male under 35
#C3: Female over 35
#C4: Male over 35


# Classes
c1 = data[(data.Sex == 'Femmina') & (data.Age <= 35)]
c2 = data[(data.Sex == 'Maschio') & (data.Age <= 35)]
c3 = data[(data.Sex == 'Femmina') & (data.Age > 35)]
c4 = data[(data.Sex == 'Maschio') & (data.Age > 35)]


# Total number of people
N1 = len(c1.Date)
N2 = len(c2.Date)
N3 = len(c3.Date)
N4 = len(c4.Date)

# Number of people of class c1 that buy item 1
nc1_i1 = c1.Date[c1.AW == 'Sì'].count()
nc2_i1 = c2.Date[c2.AW == 'Sì'].count()
nc3_i1 = c3.Date[c3.AW == 'Sì'].count()
nc4_i1 = c4.Date[c4.AW == 'Sì'].count()

# P(buy item 1) = i1
c1_i1_param = nc1_i1 / N1
c2_i1_param = nc2_i1 / N2
c3_i1_param = nc3_i1 / N3
c4_i1_param = nc4_i1 / N4

# Number of people that buy item 2 (or buy item 2 + P0)
nc1_i2_p0 = c1.Date[c1.WS0 == 'Sì'].count()
nc2_i2_p0 = c2.Date[c2.WS0 == 'Sì'].count()
nc3_i2_p0 = c3.Date[c3.WS0 == 'Sì'].count()
nc4_i2_p0 = c4.Date[c4.WS0 == 'Sì'].count()

# P(buy item 2 + P0) = P(buy item 2) = i2_p0
c1_i21_p0_param = (nc1_i2_p0) / N1
c2_i21_p0_param = (nc2_i2_p0) / N2
c3_i21_p0_param = (nc3_i2_p0) / N3
c4_i21_p0_param = (nc4_i2_p0) / N4

# Number of people that buy item 2 + P1
nc1_i2_p1 = c1.Date[c1.WS10 == 'Sì'].count()
nc2_i2_p1 = c2.Date[c2.WS10 == 'Sì'].count()
nc3_i2_p1 = c3.Date[c3.WS10 == 'Sì'].count()
nc4_i2_p1 = c4.Date[c4.WS10 == 'Sì'].count()

# P(buy item 2 + P1) = i2_p1
c1_i21_p1_param = (nc1_i2_p1 + nc1_i2_p0 ) / N1
c2_i21_p1_param = (nc2_i2_p1 + nc2_i2_p0) / N2
c3_i21_p1_param = (nc3_i2_p1 + nc3_i2_p0) / N3
c4_i21_p1_param = (nc4_i2_p1 + nc4_i2_p0) / N4

# Number of people that buy item 2 + P2
nc1_i2_p2 = c1.Date[c1.WS20 == 'Sì'].count()
nc2_i2_p2 = c2.Date[c2.WS20 == 'Sì'].count()
nc3_i2_p2 = c3.Date[c3.WS20 == 'Sì'].count()
nc4_i2_p2 = c4.Date[c4.WS20 == 'Sì'].count()

# P(buy item 2 + P2) = i2_p2
c1_i21_p2_param = (nc1_i2_p2 + nc1_i2_p1 + nc1_i2_p0) / N1
c2_i21_p2_param = (nc2_i2_p2 + nc2_i2_p1 + nc2_i2_p0) / N2
c3_i21_p2_param = (nc3_i2_p2 + nc3_i2_p1 + nc3_i2_p0) / N3
c4_i21_p2_param = (nc4_i2_p2 + nc4_i2_p1 + nc4_i2_p0) / N4

# Number of people that buy item 2 + P3
nc1_i2_p3 = c1.Date[c1.WS50 == 'Sì'].count()
nc2_i2_p3 = c2.Date[c2.WS50 == 'Sì'].count()
nc3_i2_p3 = c3.Date[c3.WS50 == 'Sì'].count()
nc4_i2_p3 = c4.Date[c4.WS50 == 'Sì'].count()

# P(buy item 2 + P3) = i2_p3
c1_i21_p3_param = (nc1_i2_p3 + nc1_i2_p2 + nc1_i2_p1 + nc1_i2_p0) / N1
c2_i21_p3_param = (nc2_i2_p3 + nc2_i2_p2 + nc2_i2_p1 + nc2_i2_p0) / N2
c3_i21_p3_param = (nc3_i2_p3 + nc3_i2_p2 + nc3_i2_p1 + nc3_i2_p0) / N3
c4_i21_p3_param = (nc4_i2_p3 + nc4_i2_p2 + nc4_i2_p1 + nc4_i2_p0) / N4

