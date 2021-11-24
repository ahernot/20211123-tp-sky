import numpy as np

POSITIVE = 1
NEGATIVE = 0


class Metrics:

    def __init__ (self, labels, predictions):
        # labels, predicitons as int

        # Labels
        labels_mask_pos = (labels == POSITIVE)
        labels_mask_neg = (labels == NEGATIVE)
        
        # Predictions
        predictions_mask_pos = (predictions == POSITIVE)
        predictions_mask_neg = (predictions == NEGATIVE)

        # Confusion matrix
        self.tpos = np.count_nonzero(np.multiply(labels_mask_pos, predictions_mask_pos))
        self.fpos = np.count_nonzero(np.multiply(labels_mask_neg, predictions_mask_pos))
        self.fneg = np.count_nonzero(np.multiply(labels_mask_pos, predictions_mask_neg))
        self.tneg = np.count_nonzero(np.multiply(labels_mask_neg, predictions_mask_neg))

        # Positive
        self.pos = self.tpos + self.fneg
        self.neg = self.fpos + self.tneg

    def __repr__ (self):
        return f'TP: {self.tpos}\nFP: {self.fpos}\nFN: {self.fneg}\nTN: {self.tneg}'

    def tpr (self): return self.tpos / self.pos 
    def tnr (self): return self.tneg / self.neg
    def fpr (self): return self.fpos / self.neg
    def fnr (self): return self.fneg / self.pos

    def precision (self): return self.ppv()
    def ppv (self):
        # Precision
        return self.tpos / (self.tpos + self.fpos)

    def fov (self):
        # False Omission Rate
        return self.fpos / (self.fpos + self.tneg)

    def error_rate (self): return self.err()
    def err (self):
        # Error rate
        return (self.fneg + self.fpos) / (self.neg + self.pos)

    def accuracy (self): return self.acc()
    def acc (self):
        return (self.tneg + self.tpos) + (self.neg + self.pos)

    def mcc (self):
        # Matthews coefficient
        num = self.tpos * self.tneg - self.fpos * self.fneg
        den = (self.tpos + self.fpos) * (self.tpos + self.fneg) * (self.tneg + self.fpos) * (self.tneg + self.fneg)
        return num / den**0.5
 
    def f_score (self, beta = 1):
        ppv = self.ppv()
        tpr = self.tpr()
        num = ppv * tpr
        den = beta**2 * ppv + tpr
        return (1 + beta**2) * num / den

    def kappa (self):
        num = 2 * (self.tpos * self.tneg - self.fneg * self.fpos)
        den = (self.tpos + self.fpos) * (self.fpos + self.tneg) + (self.tpos + self.fneg) * (self.fneg + self.tneg)
        return num / den
