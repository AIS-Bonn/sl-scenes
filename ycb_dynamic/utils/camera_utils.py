import numpy as np


class TimeDependentCamParamFunc(object):
    def __init__(self, start_val, end_val, start_t, end_t):
        self.start_val = start_val
        self.end_val = end_val
        self.start_t = start_t
        self.end_t = end_t

    def get_value(self, t):
        raise NotImplementedError


class ConstFunc(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(ConstFunc, self).__init__(start_val, end_val, start_t, end_t)

    def get_value(self, t):
        return self.start_val


class LinFunc(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(LinFunc, self).__init__(start_val, end_val, start_t, end_t)
        self.slope = (self.end_val - self.start_val) / (self.end_t - self.start_t)

    def get_value(self, t):
        return self.slope * t + self.start_val


class SinFunc(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(SinFunc, self).__init__(start_val, end_val, start_t, end_t)
        self.amp = self.end_val - self.start_val
        self.freq_mod = np.random.uniform(0.3, 0.6)

    def sin(self, x):
        return np.sin(self.freq_mod * (self.end_t - self.start_t) * x)

    def get_value(self, t):
        return self.start_val + self.amp * self.sin(t)


class LinFuncOnce(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(LinFuncOnce, self).__init__(start_val, end_val, start_t, end_t)

    def get_value(self, t):
        t_rel = (t - self.start_t) / (self.end_t - self.start_t)
        t_rel = min(1, max(0, t_rel))
        return t_rel * self.end_val + (1 - t_rel) * self.start_val


class TanhFunc(TimeDependentCamParamFunc):
    def __init__(self, start_val, end_val, start_t, end_t):
        super(TanhFunc, self).__init__(start_val, end_val, start_t, end_t)
        self.freq_mod = np.random.uniform(4, 8)

    def tanh(self, x):
        return np.tanh(self.freq_mod * x - (self.freq_mod / 2))

    def get_value(self, t):
        t_rel = (t - self.start_t) / (self.end_t - self.start_t)
        return self.start_val + (self.end_val - self.start_val) * (1 + self.tanh(t_rel))
