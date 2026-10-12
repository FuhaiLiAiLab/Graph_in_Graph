"""Shared validation-checkpoint rules."""
import math
RULES=('ba','f1','ap','balanced_min','balanced_mean','balanced_guard')
DEFINITIONS={'ba':'BA','f1':'Macro-F1','ap':'Macro-AP','balanced_min':'min(Macro-F1,Macro-AP)',
             'balanced_mean':'(Macro-F1+Macro-AP)/2','balanced_guard':'(Macro-F1+Macro-AP)/2+0.5*BA'}


def normalize_rules(rules):
    values=list(rules or [])
    if len(set(values))!=len(values) or any(r not in RULES for r in values):raise ValueError('Unknown or duplicate checkpoint rules')
    return values


def rule_score(metrics,rule):
    ba,f1,ap=(float(metrics[k]) for k in ['balanced_accuracy','macro_f1','macro_auprc'])
    if not all(math.isfinite(x) for x in [ba,f1,ap]):raise ValueError('Nonfinite checkpoint metrics')
    return {'ba':ba,'f1':f1,'ap':ap,'balanced_min':min(f1,ap),'balanced_mean':(f1+ap)/2,
            'balanced_guard':(f1+ap)/2+.5*ba}[rule]
