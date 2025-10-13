import numpy as np
import os
import tensorflow as tf
from codebase.utils import make_dir_if_not_exist

# Updated by @Xianwen, March 4
# to save training and valid set as well.
class ResultLogger(object):
    def __init__(self, dname, saver=None):
        self.dname = dname
        make_dir_if_not_exist(self.dname)
        self.ckptdir = os.path.join(self.dname, 'checkpoints')
        make_dir_if_not_exist(self.ckptdir)
        self.npzdir = os.path.join(self.dname, 'npz')
        make_dir_if_not_exist(self.npzdir)
        self.saver = saver if not saver is None else tf.train.Saver()
        # result saving files, updated by @Xianwen
        # test
        self.testcsv_name = os.path.join(self.dname, 'test_metrics.csv')
        self.testcsv = open(self.testcsv_name, 'w')
        # train
        self.traincsv_name = os.path.join(self.dname, 'train_metrics.csv')
        self.traincsv = open(self.traincsv_name, 'w')
        # valid
        self.validcsv_name = os.path.join(self.dname, 'valid_metrics.csv')
        self.validcsv = open(self.validcsv_name, 'w')

    def save_metrics(self, D, phase='test'):
        """save D (a dictionary of metrics: string to float) as csv"""
        if phase == 'test':
            for k in D:
                s = '{},{:.7f}\n'.format(k, D[k])
                self.testcsv.write(s)
            self.testcsv.close()
            print('Metrics saved to {}'.format(self.testcsv_name))
        elif phase == 'train':
            for k in D:
                s = '{},{:.7f}\n'.format(k, D[k])
                self.traincsv.write(s)
            self.traincsv.close()
            print('Metrics saved to {}'.format(self.traincsv_name))
        elif phase == 'valid':
            for k in D:
                s = '{},{:.7f}\n'.format(k, D[k])
                self.validcsv.write(s)
            self.validcsv.close()
            print('Metrics saved to {}'.format(self.validcsv_name))
        else:
            raise ValueError('Invalid phase name!')


    def save_tensors(self, D, phase='test'):
        """Updated by @Xianwen: This function saves repr tensors by the given phase."""
        for k in D:
            fname = os.path.join(self.npzdir, '{}_{}.npz'.format(k, phase))
            np.savez(fname, X=D[k])

