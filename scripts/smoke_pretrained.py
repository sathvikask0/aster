"""Real pretrained encoder integration test; synthetic labels, NOT biology training."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from huggingface_hub import model_info
from aster import AsterConfig,AsterModel
from aster.model import blocks


def main():
    torch.manual_seed(42);torch.set_num_threads(8)
    cfg=AsterConfig(cell_features=16)
    cfg.protein_revision=model_info(cfg.protein_model).sha
    cfg.text_revision=model_info(cfg.text_model).sha
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=AsterModel(cfg).to(device).train()
    pt=AutoTokenizer.from_pretrained(cfg.protein_model,revision=cfg.protein_revision)
    tt=AutoTokenizer.from_pretrained(cfg.text_model,revision=cfg.text_revision)
    # Short artificial sequences and numerical inputs exercise gradient paths only.
    seqs=['MALWMRLLPLLALLALWGPDPAAA','MSTNPKPQRKTKRNTNRRPQDVKF']
    intervention={k:v.to(device) for k,v in pt(seqs,padding=True,return_tensors='pt').items()}
    readout={k:v.to(device) for k,v in pt(seqs[::-1],padding=True,return_tensors='pt').items()}
    questions=['After reducing the target gene, the readout RNA level decreases.',
               'After reducing the target gene, the readout RNA level remains within the effect threshold.',
               'After reducing the target gene, the readout RNA level increases.']
    options={k:v.reshape(2,3,-1).to(device) for k,v in tt(questions*2,padding=True,return_tensors='pt').items()}
    cell=torch.randn(2,16,device=device);labels=torch.tensor([0,2],device=device)
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    losses=[];norms={}
    for _ in range(3):
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(intervention,readout,cell,options),labels)
        assert torch.isfinite(loss);loss.backward()
        for name,enc in [('protein',model.protein),('text',model.text)]:
            grads=[p.grad for p in blocks(enc)[-1].parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            norms[name]=sum(float(g.abs().sum().cpu()) for g in grads);assert norms[name]>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach().cpu()))
    updated={prefix:sum(int(not torch.equal(before[n],p.detach().cpu())) for n,p in model.named_parameters() if n.startswith(prefix+'.') and p.requires_grad) for prefix in ['protein','text']}
    frozen_ok=all(torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters() if not p.requires_grad)
    assert all(updated.values()) and frozen_ok
    result={'purpose':'Integration smoke test only; artificial sequences, synthetic features and labels. No biological accuracy measured.',
        'device':device,'protein_model':cfg.protein_model,'protein_revision':cfg.protein_revision,'text_model':cfg.text_model,'text_revision':cfg.text_revision,
        'protein_unfrozen_layers':cfg.protein_last_layers,'text_unfrozen_layers':cfg.text_last_layers,
        'steps':3,'losses':losses,'last_block_gradient_l1':norms,'updated_parameter_tensors':updated,'all_frozen_parameters_unchanged':frozen_ok,
        'decision':model.decide(intervention,readout,cell,options)}
    Path('reports').mkdir(exist_ok=True);Path('reports/pretrained_smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
