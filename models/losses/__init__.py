# coding:utf-8
__all__ = ['build_loss', 'build_kd_loss']

from .DB_loss import DBLoss
from .kd_loss import KdLoss

support_loss = ['DBLoss', 'KdLoss']


def build_loss(loss_name, **kwargs):
    assert loss_name in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_name)(**kwargs)
    return criterion


def build_kd_loss(loss_name, **kwargs):
    assert loss_name in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_name)(**kwargs)
    return criterion
