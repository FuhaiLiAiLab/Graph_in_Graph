"""Training-budget policy: checkpoint cadence and conservative early stopping."""
import copy
import math


def normalize_policy(value,epochs):
    raw=copy.deepcopy(value or {})
    if set(raw)-{'checkpoint_interval','scheduler_horizon','early_stopping'}:
        raise ValueError('Unknown training policy keys')
    interval=raw.get('checkpoint_interval',1)
    horizon=raw.get('scheduler_horizon',epochs)
    if type(interval) is not int or interval<1:raise ValueError('checkpoint_interval must be positive')
    if type(horizon) is not int or horizon<epochs:raise ValueError('scheduler_horizon must cover the epoch cap')
    early=dict(enabled=False,min_epochs=min(150,epochs),patience=75,min_delta=.001)
    supplied=raw.get('early_stopping',{})
    if not isinstance(supplied,dict) or set(supplied)-set(early)-{'monitor'}:raise ValueError('Invalid early_stopping keys')
    early.update(supplied)
    if early.get('monitor','selection') not in ('selection','balanced_guard'):
        raise ValueError('Invalid early stopping monitor')
    if type(early['enabled']) is not bool:raise ValueError('early_stopping.enabled must be boolean')
    if any(type(early[k]) is not int or early[k]<1 for k in ['min_epochs','patience']):raise ValueError('Invalid early stopping counts')
    if early['min_epochs']>epochs:raise ValueError('min_epochs exceeds epoch cap')
    if type(early['min_delta']) not in (int,float) or not math.isfinite(early['min_delta']) or early['min_delta']<0:
        raise ValueError('Invalid early stopping min_delta')
    return dict(checkpoint_interval=interval,scheduler_horizon=horizon,early_stopping=early)


class StopTracker:
    def __init__(self,policy,state=None):
        self.policy=dict(policy)
        self.best=None
        self.last_improvement=0
        if state is not None:
            self.best=state['best'];self.last_improvement=state['last_improvement']

    def observe(self,epoch,value):
        if not math.isfinite(value):raise ValueError('Early stopping metric must be finite')
        if self.best is None or value>self.best+self.policy['min_delta']:
            self.best=float(value);self.last_improvement=epoch
        return (self.policy['enabled'] and epoch>=self.policy['min_epochs']
                and epoch-self.last_improvement>=self.policy['patience'])

    def state_dict(self):
        return dict(best=self.best,last_improvement=self.last_improvement)
