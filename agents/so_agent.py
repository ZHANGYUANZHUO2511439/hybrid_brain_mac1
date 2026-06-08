class SOAgent:

    def __init__(self):

        self.target_low = 0.5
        self.target_high = 1.0

    def update_noise(self, noise, freq):

        # 频率太高
        if freq > self.target_high:

            noise += 0.0001

        # 频率太低
        elif freq < self.target_low:

            noise -= 0.0001

        return noise