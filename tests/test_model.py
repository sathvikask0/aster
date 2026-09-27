import pytest
import torch
from transformers import BertConfig, BertModel, EsmConfig, EsmModel
from aster import AsterConfig, AsterModel
from aster.model import blocks


def fixture():
    torch.manual_seed(0)
    config=AsterConfig(protein_last_layers=1,text_last_layers=1,cell_features=8,latent_dim=16)
    p=EsmModel(EsmConfig(vocab_size=33,hidden_size=24,num_hidden_layers=2,num_attention_heads=4,intermediate_size=48,pad_token_id=1,mask_token_id=32,token_dropout=False))
    t=BertModel(BertConfig(vocab_size=40,hidden_size=24,num_hidden_layers=2,num_attention_heads=4,intermediate_size=48))
    model=AsterModel(config,p,t)
    seq={'input_ids':torch.randint(4,25,(2,6)),'attention_mask':torch.ones(2,6,dtype=torch.long)}
    opts={'input_ids':torch.randint(4,35,(2,3,7)),'attention_mask':torch.ones(2,3,7,dtype=torch.long)}
    return model,(seq,seq,torch.randn(2,8),opts)


def test_both_suffixes_receive_updates_frozen_prefixes_do_not():
    model,args=fixture();model.train();before={k:v.detach().clone() for k,v in model.named_parameters()}
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.01)
    loss=torch.nn.functional.cross_entropy(model(*args),torch.tensor([0,2]));loss.backward()
    for enc in [model.protein,model.text]:
        grads=[p.grad for p in blocks(enc)[-1].parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
        assert all(p.grad is None for p in blocks(enc)[0].parameters())
    opt.step()
    for prefix in ['protein.','text.']:
        assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad)
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters() if not p.requires_grad)


def test_candidate_permutation_and_mask():
    model,args=fixture();model.eval();first=model(*args);order=[2,0,1]
    altered=(*args[:3],{k:v[:,order] for k,v in args[3].items()})
    assert torch.allclose(model(*altered),first[:,order],atol=1e-6)
    mask=torch.tensor([[True,False,True],[True,True,False]])
    out=model(*args,option_mask=mask).softmax(-1)
    assert torch.all(out[~mask]==0)
    assert torch.allclose(out.sum(-1),torch.ones(2))
    with pytest.raises(ValueError):model(*args,option_mask=torch.zeros(2,3,dtype=torch.bool))
    assert model.decide(*args)['calibrated'] is False


def test_checkpoint_reload(tmp_path):
    model,args=fixture();model.eval();model.save(tmp_path/'model.pt')
    other,_=fixture();ck=torch.load(tmp_path/'model.pt',weights_only=True);other.load_state_dict(ck['state_dict']);other.eval()
    assert torch.allclose(model(*args),other(*args),atol=1e-6)


def test_training_rejects_intervention_leakage():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('trainer',Path(__file__).parents[1]/'scripts/train.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    row={'split':'train','intervention_group':'same_gene','cell_state':[1.0], 'options':['up','down'],'label':0,'evidence_id':'experiment:1','intervention_sequence':'MAK','readout_sequence':'MAL'}
    with pytest.raises(ValueError,match='leakage'):
        trainer.validate_rows([row,{**row,'split':'validation'}],1)
