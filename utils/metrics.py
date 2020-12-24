# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        print('label_true, label_pred', label_true, label_pred)
        mask = (label_true >= 0) & (label_true < n_class)
        print('==mask:', mask)
        if np.sum((label_pred[mask] < 0)) > 0:
            print(label_pred[label_pred < 0])
        hist = np.bincount(n_class * label_true[mask].astype(int) +
                           label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        print('==hist:', hist)
        return hist

    def update(self, label_trues, label_preds):
        self.confusion_matrix = confusion_matrix(label_trues.flatten(), label_preds.flatten())
        # for lt, lp in zip(label_trues, label_preds):
        #     try:
        #         self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        #     except:
        #         pass
        return self.confusion_matrix
    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 0.0001)
        print('===acc:', acc)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.0001)
        print('==acc_cls:', acc_cls)
        acc_cls = np.nanmean(acc_cls)
        print('==acc_cls:', acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.0001)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum() + 0.0001)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu, }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def get_f1_score(label_trues, label_preds):
    y_true = label_trues.flatten()
    y_pred = label_preds.flatten()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1
def debug_main():
    running_metric_text = runningScore(2)
    label_trues = np.array([[1, 0, 1, 1],
                            [1, 0, 1, 1]])

    label_preds = np.array([[1, 0, 0, 1],
                            [1, 1, 0, 1]])

    # print('==label_trues:\n', label_trues)
    # print('==label_preds:\n', label_preds)
    # confusion_matrix = running_metric_text.update(label_trues, label_preds)
    # print('===confusion_matrix:\n', confusion_matrix)
    # res = running_metric_text.get_scores()
    # print('==res:', res)
    # print(res[0]['Mean Acc'])
    precision, recall, f1 = get_f1_score(label_trues, label_preds)
    print(precision)
    print(recall)
    print(f1)
if __name__ == '__main__':
    debug_main()