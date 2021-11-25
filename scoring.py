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

        # Rates
        self.tpr = self.tpos / self.pos
        self.tnr = self.tneg / self.neg
        self.fpr = self.fpos / self.neg
        self.fnr = self.fneg / self.pos

        # Other metrics
        self.ppv = self.tpos / (self.tpos + self.fpos)  # precision
        self.fov = self.fpos / (self.fpos + self.tneg)  # false omission rate
        self.err = (self.fneg + self.fpos) / (self.neg + self.pos)  # error rate
        self.acc = (self.tneg + self.tpos) / (self.neg + self.pos)  # accuracy

        # Scores
        self.f1 = self.f_score()
        self.mcc = self.__mcc()
        self.kappa = self.__kappa()
        


    def __repr__ (self):
        repr_list = [
            f'TP  = {self.tpos}',
            f'FP  = {self.fpos}',
            f'FN  = {self.fneg}',
            f'TN  = {self.tneg}',
            f'P   = {self.pos}',
            f'N   = {self.neg}',
            f'=================================',
            f'TPR = {self.tpr}',
            f'TNR = {self.tnr}',
            f'FPR = {self.fpr}',
            f'FNR = {self.fnr}',
            f'=================================',
            f'Precision           = {self.ppv}',
            f'False omission rate = {self.fov}',
            f'Error rate          = {self.err}',
            f'Accuracy            = {self.acc}',
            f'=================================',
            f'F1-score            = {self.f1}',
            f'MCC                 = {self.mcc}',
            f'Kappa score         = {self.kappa}'
        ]
        return '\n'.join(repr_list)


    def __mcc (self):
        # Matthews coefficient
        num = self.tpos * self.tneg - self.fpos * self.fneg
        den = (self.tpos + self.fpos) * (self.tpos + self.fneg) * (self.tneg + self.fpos) * (self.tneg + self.fneg)
        return num / den**0.5
 
    def f_score (self, beta = 1):
        ppv = self.ppv
        tpr = self.tpr
        num = ppv * tpr
        den = beta**2 * ppv + tpr
        return (1 + beta**2) * num / den

    def __kappa (self):
        num = 2 * (self.tpos * self.tneg - self.fneg * self.fpos)
        den = (self.tpos + self.fpos) * (self.fpos + self.tneg) + (self.tpos + self.fneg) * (self.fneg + self.tneg)
        return num / den
