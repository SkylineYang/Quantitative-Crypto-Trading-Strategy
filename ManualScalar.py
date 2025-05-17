import numpy as np

class ManualMinMaxScalar():
    '''
    Manually implement Min Max Scalar by flatten 3D dataset into 2D.
    '''
    def __init__(self):
        self.max = 1
        self.min = 0

    def reshape_3D(self, X, shapes):
        scaled_list = []
        start = 0
        for l in shapes:
            scaled_list.append(X[start: start+l])
            start += l
        return np.array(scaled_list, dtype=object)

    def fit_transform(self, X, shapes=None):
        X_flattened = np.concatenate(X, axis=0)
        self.max = X_flattened.max(axis=0)
        self.min = X_flattened.min(axis=0)
        return self.reshape_3D((X_flattened - self.min) / (self.max - self.min), shapes)

    def transform(self, X, shapes=None, reshape=True):
        if reshape:
            X_flattened = np.concatenate(X, axis=0)
            return self.reshape_3D((X_flattened - self.min) / (self.max - self.min), shapes)
        else:
            return (X - self.min) / (self.max - self.min)
    
    def inverse_transform(self, X, shapes=None, reshape=True):
        if reshape:
            X_flattened = np.concatenate(X, axis=0)
            return self.reshape_3D(X_flattened * (self.max - self.min) + self.min, shapes)
        else:
            return X * (self.max - self.min) + self.min