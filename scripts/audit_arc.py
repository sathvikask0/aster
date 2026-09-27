"""Audit official Arc sample; compute descriptive leave-intervention-out baselines."""
import argparse, hashlib, json
from pathlib import Path
import urllib.request
import numpy as np
import anndata as ad
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

COMMIT = '5e64833518a6603a0301cbe28185d49c30f4a986'
URL = f'https://raw.githubusercontent.com/ArcInstitute/cell-eval2/{COMMIT}/docs/data/H1-VCC-2025-training.h5ad'


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/arc_sample.h5ad');p.add_argument('--out',default='reports/arc_sample_audit.json');a=p.parse_args()
    path=Path(a.data);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(URL,path)
    expected='eb36c766cbf76353f9981cb3a3aa32137622d1de53b29d861c483742bcd4dec7'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Sample checksum mismatch; do not reuse altered data')
    x=ad.read_h5ad(path)
    matrix=x.X.toarray() if hasattr(x.X,'toarray') else np.asarray(x.X)
    assert np.isfinite(matrix).all() and (matrix>=0).all()
    assert x.obs_names.is_unique and x.var_names.is_unique
    assert not x.obs.target_gene.isna().any()
    assert np.allclose(matrix,np.round(matrix))
    groups=x.obs.target_gene.astype(str).to_numpy();control=groups=='non-targeting'
    assert control.any()
    # Subset-only normalization is exploratory; full data must use full library totals.
    totals=matrix.sum(1);assert (totals>0).all()
    normalized=matrix/totals[:,None]*10000
    ref=normalized[control].mean(0)
    eligible=ref>=1
    targets=sorted(set(groups)-{'non-targeting'})
    effects=np.array([np.log2((normalized[groups==t].mean(0)+1)/(ref+1)) for t in targets])[:,eligible]
    # Operational observed-effect bins, not significance or equivalence tests.
    labels=np.where(effects<-.5,0,np.where(effects>.5,2,1))
    results=[]
    for i,t in enumerate(targets):
        train=np.delete(labels,i,axis=0)
        prior=np.bincount(train.ravel(),minlength=3)+1
        prior=prior/prior.sum()
        gene_prior=np.stack([(train==c).sum(0)+1 for c in range(3)],-1)
        gene_prior=gene_prior/gene_prior.sum(-1,keepdims=True)
        for name,prob in [('global_train_prior',np.tile(prior,(labels.shape[1],1))),('readout_gene_train_prior',gene_prior)]:
            y=labels[i];pred=prob.argmax(1)
            results.append({'held_out_intervention':t,'model':name,'rows':len(y),
                'accuracy':float((pred==y).mean()),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
                'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
                'log_loss':float(log_loss(y,prob,labels=[0,1,2])),
                'brier':float(np.mean(np.sum((prob-np.eye(3)[y])**2,axis=1)))})
    result={'source_url':URL,'source_commit':COMMIT,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'cells':x.n_obs,'measured_genes':x.n_vars,'groups':x.obs.target_gene.value_counts().to_dict(),
        'batches':x.obs.batch.value_counts().to_dict(),'batch_count':int(x.obs.batch.nunique()),'unique_cell_ids':x.obs_names.is_unique,'unique_gene_ids':x.var_names.is_unique,
        'missing_obs':x.obs.isna().sum().to_dict(),'nonnegative_finite_integer_counts':True,
        'eligible_readout_genes':int(eligible.sum()),'label_counts':np.bincount(labels.ravel(),minlength=3).tolist(),
        'label_order':['decrease','within_threshold','increase'],'log2_effect_threshold':0.5,
        'folds':results,'scope':'Descriptive pipeline check only; five interventions, one cell line, subset-normalized counts. No biological generalization or calibrated probability claim.',
        'limitations':['Baseline pools experimental batches; batch composition can confound descriptive differences.','Cells are subsamples, not independent experimental replicates.','No full library totals: normalization uses only the 1000 supplied genes.','Within-threshold is not evidence of no biological effect.','No significance labels or donor-level uncertainty can be established from this subset.']}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['cells','measured_genes','groups','eligible_readout_genes','label_counts','scope']},indent=2))

if __name__=='__main__':main()
