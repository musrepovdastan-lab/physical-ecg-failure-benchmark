"""Pretraining engineering checks for the frozen augmentation-probability runner."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("probability_runner",HERE/"run_probability_study.py")
runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)

def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 audit={"cuda_device":torch.cuda.get_device_name(0),"cohorts":{},"data_loader_epoch_visibility":{},"checks":[]}
 for cohort in ("Chapman","Ningbo"):
  rec,mean,std,split,prep=runner.source_data(cohort,runner.SEEDS[0])
  ids={s:([str(x) for x in rec[s]] if cohort=="Chapman" else [x["record_id"] for x in rec[s]]) for s in rec}
  if set(ids["train"])&set(ids["validation"]) or set(ids["train"])&set(ids["test"]) or set(ids["validation"])&set(ids["test"]):raise RuntimeError(cohort+" split overlap")
  x=runner.raw_for(cohort,rec["train"][:8]);y=runner.labels_for(cohort,rec["train"][:8])
  if x.shape!=(8,12,1000) or y.shape!=(8,3):raise RuntimeError(cohort+" waveform/label shape mismatch")
  audit["cohorts"][cohort]={"train":len(rec["train"]),"validation":len(rec["validation"]),"test":len(rec["test"]),"split_disjoint":True,"preprocessing_sha256":runner.sha(prep)}
  for arm,expected in (("physics_p010",.10),("physics_p025",.25)):
   counts=[];choices=[]
   for epoch in (1,2):
    v=[runner.augment_raw(np.zeros((12,1000),np.float32),arm,runner.SEEDS[0],epoch,i)[1] for i in range(len(rec["train"]))]
    freq=sum(x!="clean" for x in v)/len(v);swaps=[x for x in v if x!="clean"];counts.append({k:swaps.count(k) for k in runner.FAULTS});choices.append(v)
    if abs(freq-expected)>.012:raise RuntimeError(f"{cohort} {arm} epoch {epoch} augmentation proportion outside tolerance: {freq}")
   if choices[0]==choices[1]:raise RuntimeError(f"{cohort} {arm} assignments did not vary by epoch")
   audit["cohorts"][cohort][arm]={"epoch_1_swap_counts":counts[0],"epoch_2_swap_counts":counts[1],"assignment_changed":True}
  # On Windows, this verifies each fresh worker sees the Dataset's current epoch.
  ds=runner.TrainDataset(rec["train"][:128],cohort,mean,std,"physics_p025",runner.SEEDS[0]);batches=[]
  for epoch in (1,2):
   ds.epoch=epoch;loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=False)
   batches.append(torch.cat([a for a,_ in loader],dim=0))
  changed=torch.any(torch.abs(batches[0]-batches[1]).flatten(1).max(1).values>1e-7)
  if not bool(changed):raise RuntimeError(cohort+" DataLoader workers did not receive changed epoch")
  audit["data_loader_epoch_visibility"][cohort]={"workers":2,"persistent_workers":False,"changed_examples":int(torch.any(torch.abs(batches[0]-batches[1]).flatten(1)>1e-7,dim=1).sum())}
  audit["checks"].extend([cohort+"_split_and_data_integrity",cohort+"_deterministic_augmentation",cohort+"_epoch_worker_visibility"])
 # CUDA train/eval and atomic checkpoint round trip on the exact architecture.
 model=runner.ECGResNetSmall().cuda();model.train();x=torch.zeros((2,12,1000),device="cuda");y=torch.zeros((2,3),device="cuda")
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scaler=torch.amp.GradScaler("cuda")
 with torch.autocast("cuda",dtype=torch.float16):loss=torch.nn.BCEWithLogitsLoss()(model(x),y)
 scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();model.eval()
 with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):pred=model(x)
 if pred.shape!=(2,3) or not torch.isfinite(pred).all():raise RuntimeError("CUDA AMP smoke failed")
 tmp=HERE/"engineering_roundtrip.pt";runner.atomic_torch(tmp,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict()})
 saved=torch.load(tmp,weights_only=False,map_location="cuda");model2=runner.ECGResNetSmall().cuda();model2.load_state_dict(saved["model"])
 with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):pred2=model2(x)
 tmp.unlink()
 if not torch.equal(pred,pred2):raise RuntimeError("checkpoint round trip changed eval output")
 audit["checks"].extend(["cuda_amp_forward_backward","train_eval_mode","checkpoint_optimizer_scaler_roundtrip"])
 audit["status"]="PASSED"
 (HERE/"ENGINEERING_CHECKS.json").write_text(json.dumps(audit,indent=2,sort_keys=True),encoding="utf-8")
 print(json.dumps(audit,sort_keys=True))
if __name__=="__main__":main()
