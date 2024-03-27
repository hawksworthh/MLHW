import numpy as np

class CoubDouglasFunction:

    def x_y(self, data, need_lin = False):
        x = data[:,:-1]
        if not need_lin:
            x = np.c_[np.ones(x.shape[0]), x]
        y = data[:,-1].reshape(-1, 1)
        return (x, y)

    def hardness_func(self, data, threshold):
        D = data.shape[-1]
        N = data.shape[0]
        if D ** 3 + D ** 2 * N >= threshold:
            return 1
        else:
            return 0
        
    def gradient_boost_optimizer(self, x, y, alpha, basic_tries:100):
        N = x.shape[0]
        start_point = np.random.rand(1, x.shape[-1]).reshape(-1, 1)
        for i in range(basic_tries):
            function = np.dot(x, start_point)
            error = function - y
            gradient = 2 * np.dot(x.T, error) / N
            start_point -= alpha * gradient
        return start_point

    def mathematical_optimization(self, x, y):
        return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)