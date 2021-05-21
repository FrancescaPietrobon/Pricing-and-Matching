class CUMSUM:
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return 0
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            # TODO I left this whole file as it was, except for adding this two ".any()" (same result with ".all()")
            #  because it has to do the max between 0 and an array (weirdly it happens only with CUMCUM_UCB1_item1)
            self.g_plus = max(0, (self.g_plus + s_plus).any())
            self.g_minus = max(0, (self.g_minus + s_minus).any())
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0
