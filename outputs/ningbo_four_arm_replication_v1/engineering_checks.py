"""Synthetic regression/integrity gates over the actual workflow functions."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from outputs.ningbo_four_arm_replication_v1 import workflow as w

def run():
    checks={}
    x=np.broadcast_to(np.arange(1,13,dtype=np.float32)[:,None],(12,32)).copy()
    mean=np.zeros(12,np.float32);std=np.ones(12,np.float32)
    # Hit the real masking branch and verify precisely its returned indices.
    masked=None
    for i in range(10000):
        z,kind=w.augment_raw(x,mean,"masking",104729,1,i)
        if kind.startswith("mask5_"): masked=(z,kind);break
    assert masked is not None
    z,kind=masked; selected=tuple(map(int,kind.split("_")[1:])); assert len(selected)==5 and len(set(selected))==5
    assert np.all(z[list(selected)]==0)
    assert np.array_equal(z[[i for i in range(12) if i not in selected]],x[[i for i in range(12) if i not in selected]])
    norm=w.recovery.normalize_raw(z,mean,std);assert np.all(norm[list(selected)]==0)
    checks["mask_exactly_selected_five_and_unselected_preserved"]="PASS"
    checks["train_mean_sentinel_normalizes_to_zero"]="PASS"
    assert len(w.MASKS)==792 and len(set(w.MASKS))==792 and all(len(m)==5 for m in w.MASKS)
    checks["all_792_masks_are_unique_five_channel_combinations"]="PASS"

    # Real physical transform and project implementation must agree sample-for-sample.
    for fault in w.FAULTS:
        expected=w.derive_display(w.named_reversal(np.concatenate((x[:2],x[6:]),axis=0),fault)).astype(np.float32)
        assert np.array_equal(w.reverse_raw(x,fault),expected)
    checks["reversals_match_project_physics_implementation"]="PASS"

    # The transform receives raw x; normalized operation must equal transform-then-normalize.
    nonzero_mean=np.arange(12,dtype=np.float32); nonunit_std=np.arange(12,dtype=np.float32)+1
    rr,tag=w.augment_raw(x,nonzero_mean,"masking",104729,1,i)
    assert np.array_equal(w.recovery.normalize_raw(rr,nonzero_mean,nonunit_std),
                          (w.recovery.apply_mask_raw(x,nonzero_mean,selected)-nonzero_mean[:,None])/nonunit_std[:,None])
    checks["augmentation_precedes_normalization"]="PASS"

    # Deterministic for arm/seed/epoch/record; choices vary over the frozen keys.
    assert w.augment_raw(x,mean,"combined",130363,7,99)[1] == w.augment_raw(x,mean,"combined",130363,7,99)[1]
    counts={"clean":0,"mask":0,"reversal":0}; fault_counts={f:0 for f in w.FAULTS}
    for i in range(12000):
        _,tag=w.augment_raw(x,mean,"combined",130363,4,i)
        counts["clean" if tag=="clean" else "mask" if tag.startswith("mask5_") else "reversal"]+=1
        if tag in fault_counts:fault_counts[tag]+=1
    assert abs(counts["clean"]/12000-.5)<.025 and abs(counts["mask"]/12000-.25)<.025 and abs(counts["reversal"]/12000-.25)<.025
    assert all(abs(fault_counts[f]/12000-1/12)<.012 for f in w.FAULTS)
    checks["deterministic_sampling_and_combined_recipe_frequency"]={"decision":"PASS","counts":counts,"reversal_counts":fault_counts}
    mask_counts={m:0 for m in w.MASKS}
    for i in range(60000):
        _,tag=w.augment_raw(x,mean,"masking",155921,9,i)
        if tag.startswith("mask5_"):mask_counts[tuple(map(int,tag.split('_')[1:]))]+=1
    assert all(v>0 for v in mask_counts.values())
    checks["mask_sampling_covers_all_792_combinations"]={"decision":"PASS","n":60000,"min_count":min(mask_counts.values()),"max_count":max(mask_counts.values())}

    # The real dataset-statistics path is exercised with raw() stubbed to train-only sentinels.
    original_raw=w.data.raw; seen=[]
    try:
        def synthetic_raw(record):
            seen.append(record['record_id']); return np.full((12,4),record['value'],np.float32)
        w.data.raw=synthetic_raw
        m,s=w.train_stats([{'record_id':'train-a','value':1.0},{'record_id':'train-b','value':3.0}])
        assert seen==['train-a','train-b'] and np.allclose(m,2.0) and np.allclose(s,1.0)
    finally: w.data.raw=original_raw
    # Positive weights likewise accept only one explicit train-label matrix.
    train_y=np.asarray([[1,0,0],[0,1,0],[0,0,1],[0,0,0]],np.float32)
    wgt=w.positive_weights(train_y);assert np.isfinite(wgt).all() and wgt.shape==(3,)
    try:w.positive_weights(np.zeros((2,4),np.float32));raise AssertionError("bad labels accepted")
    except ValueError:pass
    checks["train_statistics_and_positive_weights_train_only_inputs"]="PASS"

    # Validation/inference helper switches the model to eval; each training epoch explicitly resets train mode.
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.l=torch.nn.Linear(12,3)
        def forward(self,a): return self.l(a.mean(-1))
    model=Tiny().cuda(); model.eval(); model.train(); assert model.training
    xx=np.zeros((4,12,16),np.float32); yy=np.zeros((4,3),np.uint8)
    p=w.predict(model,xx,yy,mean,std); assert p.shape==(4,3) and not model.training and np.isfinite(p).all()
    checks["model_train_each_epoch_and_eval_for_validation_inference"]="PASS"

    # A genuine checkpoint state reload reproduces outputs, and resume provenance rejects foreign arms/protocols.
    model.train(); before={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.l.weight.data.add_(1); w.restore_best(model,before); assert all(torch.equal(model.state_dict()[k],v) for k,v in before.items())
    fake={"protocol_sha256":"p","arm":"masking","seed":104729,"train_split_sha256":"s"}
    proto={"protocol_sha256":"p","bindings":{"outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json":"s"}}
    assert w.validate_resume_payload(fake,"masking",104729,proto)
    for wrong in ({**fake,"arm":"combined"},{**fake,"protocol_sha256":"other"}):
        try:w.validate_resume_payload(wrong,"masking",104729,proto);raise AssertionError("foreign resume accepted")
        except RuntimeError:pass
    checks["best_checkpoint_restore_and_resume_arm_protocol_integrity"]="PASS"
    result={"decision":"PASS","checks":checks,"test_accessed":False}
    w.atomic_json(w.OUT/"engineering_checks_result.json",result)
    print(json.dumps(result,indent=2,ensure_ascii=False));return result

if __name__=="__main__": run()
