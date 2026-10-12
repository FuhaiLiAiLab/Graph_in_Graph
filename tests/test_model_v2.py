"""Behavioral checks for independently parameterized v2 models and legacy compatibility."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from runtime import build_model,default_config,forward,training_class_weights
from test_runtime import synthetic_data


def cfg2(**changes):
    return dict(default_config(),model_schema_version=2,**changes)


@pytest.mark.parametrize('heads,depth',[(1,1),(2,2),(4,4)])
def test_variable_heads_depth_updates_gene_assignment(heads,depth):
    cfg=cfg2(gene_hidden_dim=12,gene_output_dim=12,gig_hidden_dim=24,gig_output_dim=24,
             gene_num_head=heads,gig_num_head=heads,gene_layers=depth,patient_layers=depth)
    model=build_model(cfg,4,'cpu')
    data=synthetic_data()
    model.train(); before=model.node_weight_assign.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    from enc.geo_gigtransformer import GeneTransformerConv, PatientTransformerConv
    calls={'gene':0,'patient':0}
    def count_gene(*_): calls['gene']+=1
    def count_patient(*_): calls['patient']+=1
    hooks=[]
    for module in model.modules():
        if isinstance(module,GeneTransformerConv):hooks.append(module.register_forward_hook(count_gene))
        elif isinstance(module,PatientTransformerConv):hooks.append(module.register_forward_hook(count_patient))
    logits=forward(model,data,'cpu',2)[1]
    for hook in hooks:hook.remove()
    assert calls=={'gene':depth*2,'patient':depth}
    assert logits.shape==(3,3)
    loss=model.loss(logits,torch.tensor(data['labels_onehot'])[None],torch.tensor(data['gene_edge_index']),
                    class_weights=training_class_weights(data['labels'],'cpu'))
    loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before,model.node_weight_assign)
    model.eval()
    torch.testing.assert_close(forward(model,data,'cpu',1)[1],forward(model,data,'cpu',3)[1])


def test_dropout_changes_training_outputs_but_not_evaluation():
    model=build_model(cfg2(gene_attention_dropout=.4,representation_dropout=.4),4,'cpu')
    data=synthetic_data();model.train()
    torch.manual_seed(1); a=forward(model,data,'cpu')[1]
    torch.manual_seed(2); b=forward(model,data,'cpu')[1]
    assert not torch.equal(a,b)
    model.eval();a=forward(model,data,'cpu')[1];b=forward(model,data,'cpu')[1]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_invalid_total_width_rejected():
    with pytest.raises(ValueError,match='divisible'):
        build_model(cfg2(gene_num_head=2,gene_output_dim=9),4,'cpu')


def test_legacy_patient_graph_cannot_disable_only_subject_path():
    with pytest.raises(ValueError,match='root'):
        build_model(cfg2(patient_root_weight=False),4,'cpu')


def test_data_dimensions_not_arbitrary_search_knobs():
    with pytest.raises(ValueError,match='classes'):
        build_model(cfg2(num_classes=4),4,'cpu')


def test_class_weight_configuration_uses_training_counts():
    weights=training_class_weights([0,0,1,2,2,2],'cpu',frequency_exponent=0,majority_multiplier=1)
    torch.testing.assert_close(weights,torch.ones(3))


def test_new_config_round_trip(tmp_path):
    from runtime import read_config
    import json
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg2(gene_layers=4,optimizer='adamw')))
    loaded=read_config(p)
    assert loaded['gene_layers']==4 and loaded['optimizer']=='adamw'
    model=build_model(loaded,4,'cpu')
    saved=tmp_path/'model.pt';torch.save(model.state_dict(),saved)
    restored=build_model(loaded,4,'cpu');restored.load_state_dict(torch.load(saved,weights_only=True))
    model.eval();restored.eval()
    torch.testing.assert_close(forward(model,synthetic_data(),'cpu')[1],forward(restored,synthetic_data(),'cpu')[1])
