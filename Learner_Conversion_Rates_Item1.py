import numpy as np
np.random.seed(1234)


class Learner_Conversion_Rates_item1:
    def __init__(self):
        self.collected_samples = [[] for _ in range(4)]

    def update_conversion_rates(self, sample):
        # Appending the new non-zero conversion rates of the new sample to the list
        for class_type in range(4):
            if sample[class_type] != 0:
                self.collected_samples[class_type].append(sample[class_type])

        # Computing the mean of the list, putting 0 if no sample is present in that cell
        empirical_means = np.zeros(4)
        for class_type in range(4):
            if len(self.collected_samples[class_type]) > 0:
                empirical_means[class_type] = np.mean(self.collected_samples[class_type])
            else:
                empirical_means[class_type] = 0

        return empirical_means
