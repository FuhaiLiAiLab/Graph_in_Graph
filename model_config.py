"""Versioned configuration validation; legacy defaults retain their original behavior."""
import copy
import math

V2_DEFAULTS = dict(model_schema_version=2, gene_layers=3, patient_layers=3,
    gene_attention_dropout=0., patient_attention_dropout=0., representation_dropout=0.,
    activation_negative_slope=.5, gene_root_weight=True, patient_root_weight=True,
    gene_beta=False, patient_beta=False, optimizer='adam', beta1=.8, beta2=.999,
    optimizer_eps=1e-7, scheduler='multistep', class_frequency_exponent=1.,
    majority_class_multiplier=.5, class_weighting='power', graph_policy='legacy_directed')


def resolve_model_config(raw, base=None):
    if base is None:
        try:
            from .runtime import default_config
        except ImportError:
            from runtime import default_config
        base = default_config()
    if not isinstance(raw, dict):
        raise ValueError('Model configuration must be a dictionary')
    unknown = set(raw) - set(base) - set(V2_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    cfg = copy.deepcopy(dict(base, **raw))
    explicit_version = raw.get('model_schema_version')
    if explicit_version is not None and (type(explicit_version) is not int or explicit_version not in (1,2)):
        raise ValueError('Unsupported model_schema_version')
    extended = bool(set(raw) & (set(V2_DEFAULTS)-{'model_schema_version'}))
    if explicit_version == 1 and extended:
        raise ValueError('Extended parameters require model_schema_version=2')
    v2 = explicit_version == 2 or extended
    if v2:
        cfg = dict(V2_DEFAULTS, **cfg)
        cfg['model_schema_version'] = 2
    for key in ('gene_input_dim','gene_hidden_dim','gene_output_dim','gene_num_top_feature',
                'gig_input_dim','gig_input_transform_dim','gig_hidden_dim','gig_output_dim',
                'gene_num_head','gig_num_head'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['num_classes'] != 3 or type(cfg['num_classes']) is not int:
        raise ValueError('Expected exactly three classes')
    if cfg['graph_opt'] not in ('GinG','gene','subject'):
        raise ValueError('Unknown graph_opt')
    for key in ('lr','weight_decay','class_weight','ortho_weight','link_weight','ent_weight','gamma'):
        value=cfg[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid {key}')
    if cfg['lr'] == 0 or cfg['class_weight'] == 0 or not 0 < cfg['gamma'] <= 1:
        raise ValueError('lr/class_weight must be positive; gamma must be in (0,1]')
    if cfg['clip'] is not None and (type(cfg['clip']) not in (int,float) or not math.isfinite(cfg['clip']) or cfg['clip']<=0):
        raise ValueError('clip must be positive or None')
    milestones=cfg['milestones']
    if not isinstance(milestones,list) or any(type(x) is not int or x<=0 for x in milestones) or milestones!=sorted(set(milestones)):
        raise ValueError('milestones must be increasing distinct positive integers')
    if not v2:
        if cfg['gene_num_head'] != 1 or cfg['gig_num_head'] != 1:
            raise ValueError('Legacy configuration requires single-head gene and patient layers')
        return cfg
    if cfg['class_weighting'] not in ('none','inverse_frequency','power'):
        raise ValueError('Unknown class_weighting mode')
    if cfg['class_weighting']=='none':
        cfg['class_frequency_exponent']=0.;cfg['majority_class_multiplier']=1.
    elif cfg['class_weighting']=='inverse_frequency':
        cfg['class_frequency_exponent']=1.;cfg['majority_class_multiplier']=1.
    if cfg['graph_policy'] not in ('legacy_directed', 'patient_category_bidirectional'):
        raise ValueError('Unknown graph_policy')
    for width,heads in [('gene_hidden_dim','gene_num_head'),('gene_output_dim','gene_num_head'),
                        ('gig_hidden_dim','gig_num_head'),('gig_output_dim','gig_num_head')]:
        if cfg[width] % cfg[heads]:
            raise ValueError(f'{width} must be divisible by {heads}; widths are total after concatenation')
    for key in ('gene_layers','patient_layers'):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 8:
            raise ValueError(f'{key} must be integer 1..8')
    for key in ('gene_attention_dropout','patient_attention_dropout','representation_dropout',
                'activation_negative_slope','class_frequency_exponent','majority_class_multiplier','beta1','beta2','optimizer_eps'):
        value=cfg[key]
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError(f'Invalid {key}')
    for key in ('gene_attention_dropout','patient_attention_dropout','representation_dropout','beta1','beta2'):
        if not 0 <= cfg[key] < 1: raise ValueError(f'{key} must be in [0,1)')
    if not 0 <= cfg['activation_negative_slope'] <= 1 or not 0 <= cfg['class_frequency_exponent'] <= 1:
        raise ValueError('Slope and class exponent must be in [0,1]')
    if not 0 < cfg['majority_class_multiplier'] <= 1 or cfg['optimizer_eps']<=0:
        raise ValueError('Invalid majority multiplier or optimizer_eps')
    for key in ('gene_root_weight','patient_root_weight','gene_beta','patient_beta'):
        if type(cfg[key]) is not bool: raise ValueError(f'{key} must be bool')
    for branch in ('gene','patient'):
        if cfg[f'{branch}_beta'] and not cfg[f'{branch}_root_weight']:
            raise ValueError('beta requires root_weight')
    if cfg['graph_opt'] != 'gene' and not cfg['patient_root_weight']:
        raise ValueError('patient_root_weight cannot be disabled on legacy zero-incoming subject graph')
    if cfg['optimizer'] not in ('adam','adamw') or cfg['scheduler'] not in ('none','multistep','cosine'):
        raise ValueError('Unknown optimizer/scheduler')
    return cfg
