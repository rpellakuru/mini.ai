import numpy as np
import matplotlib.pyplot as plt

def load_data():
    X = np.load("anamoly_detection/data/X_part1.npy")
    X_val = np.load("anamoly_detection/data/X_val_part1.npy")
    y_val = np.load("anamoly_detection/data/y_val_part1.npy")
    return X, X_val, y_val


class Anamoly:
    def estimate_guassian(self, X):
        mu = np.mean(X, axis=0)
        var = np.mean((X - mu) ** 2, axis=0)
        return mu, var

    def run_anamoly(self):
        # The first step is to fit a model to the data distribution
        # Estimate the Guassian distribution for each feature xi
        pass

    def cal_multivariant_guassian(self, X, mu, var):
        print(f'Mean shape: {mu.shape}, var shape: {var.shape}')
        k = len(mu)
        if var.ndim == 1:
            var = np.diag(var)
        x_centered = X - mu
        p = (2 * np.pi) ** (-k/2) * np.linalg.det(var)**(-0.5) * \
            np.exp(-0.5 * np.sum(np.matmul(x_centered, np.linalg.pinv(var)) * x_centered, axis=1))
        
        
        return p
    
    def select_thresold(self, p_val, y_val):
        epsilon = 0
        F1 = 0
        # print(f'yval: {y_val}, p_val: {p_val}')
        step_size = (max(p_val) - min(p_val)) / 1000
        
        for i in np.arange(min(p_val), max(p_val), step_size):
            #print(f"Trying epsilon: {i}")
            p_val_groud_truth_not_anamoly = p_val[y_val == 0]
            p_val_groud_truth_anamoly = p_val[y_val == 1]
            tp = len(p_val_groud_truth_anamoly[p_val_groud_truth_anamoly <= i])
            fp = len(p_val_groud_truth_not_anamoly[p_val_groud_truth_not_anamoly <= i])
            fn = len(p_val_groud_truth_anamoly[p_val_groud_truth_anamoly > i])
            #print(f"tp: {tp}, fp: {fp}, fn: {fn}")
            precision = tp/(tp + fp)
            recall = tp/(tp + fn)
            score = (2 * precision * recall)/(precision + recall)
            if score > F1:
                F1 = score
                epsilon = i

            if score > F1:
                F1 = score
                epsilon = i
            
        return epsilon, F1





anamoly = Anamoly()
X_train, x_val, y_val = load_data()


# Compute the mean and variance for all features
m, v = anamoly.estimate_guassian(X_train)
print("Mean of each feature:", m)
print("Variance of each feature:", v)

# compute PDF on validation dataset
pdf_val = anamoly.cal_multivariant_guassian(x_val, m, v)

# Select the epsilon by calculating the F1 score
epsilon, F1 = anamoly.select_thresold(pdf_val, y_val)
print(f"Selected epsilon: {epsilon:e}, F1 Score: {F1:f}")

# Now that we have the epsilon, let's find the outliers in our dataset
# Compute the Probability Distribution Function for all features (dimensions)
pdf = anamoly.cal_multivariant_guassian(X_train, m, v)

outliers = X_train[pdf < epsilon]
print(f"Outlier points: {outliers}")
