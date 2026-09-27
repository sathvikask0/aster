"""Train Aster jointly on explicitly split, experimentally labelled JSONL rows.

This trainer does not infer assay labels or turn absent measurements into negatives.
"""
import argparse,json,random
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import model_info
from aster import AsterConfig,AsterModel


def validate_rows(rows, cell_features):
    if not rows:raise ValueError('No rows')
    groups={s:set() for s in ['train','validation','test']}
    for r in rows:
        if r['split'] not in groups:raise ValueError('Invalid split')
        groups[r['split']].add(r['intervention_group'])
        if len(r['cell_state'])!=cell_features or not np.isfinite(r['cell_state']).all():raise ValueError('Invalid cell-state vector')
        if len(r['options'])<2 or not 0<=r['label']<len(r['options']):raise ValueError('Invalid choice label')
        if not r.get('evidence_id'):raise ValueError('Experimental provenance evidence_id is required')
        for key in ['intervention_sequence','readout_sequence']:
            if not r[key] or set(r[key])-set('ACDEFGHIKLMNPQRSTVWYBXZUO'):raise ValueError('Invalid protein sequence')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if groups[a]&groups[b]:raise ValueError(f'Intervention leakage between {a} and {b}')
    if not groups['train'] or not groups['validation']:raise ValueError('Need training and validation interventions')
    return groups


def main():
    p=argparse.ArgumentParser();p.add_argument('jsonl');p.add_argument('--out',default='checkpoints/run');p.add_argument('--config');p.add_argument('--epochs',type=int,default=10);p.add_argument('--seed',type=int,default=0);p.add_argument('--encoder-lr',type=float,default=1e-5);p.add_argument('--head-lr',type=float,default=1e-4);p.add_argument('--max-protein-tokens',type=int,default=1024);p.add_argument('--max-text-tokens',type=int,default=512);a=p.parse_args()
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(8)
    rows=[json.loads(line) for line in Path(a.jsonl).read_text().splitlines() if line.strip()]
    cfg=AsterConfig(**json.loads(Path(a.config).read_text())) if a.config else AsterConfig(cell_features=len(rows[0]['cell_state']))
    groups=validate_rows(rows,cfg.cell_features)
    cfg.protein_revision=cfg.protein_revision or model_info(cfg.protein_model).sha
    cfg.text_revision=cfg.text_revision or model_info(cfg.text_model).sha
    pt=AutoTokenizer.from_pretrained(cfg.protein_model,revision=cfg.protein_revision);tt=AutoTokenizer.from_pretrained(cfg.text_model,revision=cfg.text_revision)
    def encode(r):
        seq=[]
        for k in ['intervention_sequence','readout_sequence']:
            t=pt(r[k],return_tensors='pt')
            if t['input_ids'].shape[1]>a.max_protein_tokens:raise ValueError('Protein exceeds configured limit; supply a justified domain or implement windowing. No silent truncation.')
            seq.append(t)
        texts=[r['question']+' Answer: '+v for v in r['options']]
        t=tt(texts,padding=True,return_tensors='pt')
        if t['input_ids'].shape[1]>a.max_text_tokens:raise ValueError('Question exceeds text limit')
        return (*seq,torch.tensor([r['cell_state']],dtype=torch.float32),{k:v.unsqueeze(0) for k,v in t.items()})
    # Cache tokens only, never final encoder embeddings.
    encoded=[encode(r) for r in rows]
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='mps':torch.mps.set_per_process_memory_fraction(.65)
    model=AsterModel(cfg).to(device)
    encoder=[v for n,v in model.named_parameters() if v.requires_grad and n.startswith(('protein.','text.'))]
    head=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith(('protein.','text.'))]
    opt=torch.optim.AdamW([{'params':encoder,'lr':a.encoder_lr},{'params':head,'lr':a.head_lr}])
    def args(i):
        return tuple({k:v.to(device) for k,v in x.items()} if isinstance(x,dict) else x.to(device) for x in encoded[i])
    train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation'];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    best=float('inf');history=[];patience=0
    for epoch in range(a.epochs):
        random.shuffle(train);model.train();losses=[]
        for i in train:
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(*args(i)),torch.tensor([rows[i]['label']],device=device))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            # Macro-average interventions, not readout rows, for checkpoint selection.
            by_group={}
            for i in val:
                loss=float(torch.nn.functional.cross_entropy(model(*args(i)),torch.tensor([rows[i]['label']],device=device)).cpu())
                by_group.setdefault(rows[i]['intervention_group'],[]).append(loss)
            score=float(np.mean([np.mean(v) for v in by_group.values()]))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_intervention_macro_loss':score});print(history[-1],flush=True)
        if score<best:
            best=score;patience=0;model.save(out/'best.pt')
        else:patience+=1
        (out/'training.json').write_text(json.dumps({'config':asdict(cfg),'history':history,'seed':a.seed,'split_interventions':{k:sorted(v) for k,v in groups.items()},'calibrated':False},indent=2))
        if patience>=3:break
    print('Saved validation-selected checkpoint. Test rows were not evaluated or used for selection.')

if __name__=='__main__':main()
