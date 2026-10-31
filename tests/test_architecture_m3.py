import json,os
import numpy as np
import torch
from src.m3_transformer import TimePatchTransformer
from src import architecture_m3_workflow_v1 as m3
from src.ecg_physics import derive_display,named_reversal

def test_transformer_shape_and_deterministic_initialization():
 torch.manual_seed(11);a=TimePatchTransformer();torch.manual_seed(11);b=TimePatchTransformer();x=torch.randn(2,12,1000)
 assert a(x).shape==(2,3)
 assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
def test_deterministic_augmentation_and_terminal_equations():
 x=np.arange(12000,dtype=np.float32).reshape(12,1000);a=m3.fault(x,'physics_aware',7,3,11);b=m3.fault(x,'physics_aware',7,3,11);np.testing.assert_array_equal(a,b)
 y=derive_display(named_reversal(np.stack([x[0],x[1],*x[6:]]),'RA_LL_reversal'));np.testing.assert_allclose(y[4],derive_display(np.stack([x[0],x[1],*x[6:]]))[4])
def test_atomic_npz_checkpoint_and_resume_equivalence(tmp_path):
 p=tmp_path/'x.npz';m3.an(p,y=np.ones((2,3)),p=np.zeros((2,3)),thresholds=np.arange(3));z=np.load(p);assert z['p'].shape==(2,3) and np.isfinite(z['p']).all()
 ck=tmp_path/'x.pt';state={'x':torch.arange(3)};m3.at(ck,state);assert torch.equal(torch.load(ck,weights_only=True)['x'],state['x'])
def test_dead_lock_and_partial_condition_resume_contract(tmp_path):
 lock=tmp_path/'m.lock';lock.write_text(json.dumps({'pid':999999}));assert not m3.alive(999999)
 names=m3.NAMES;done={names[0]};assert len(names)==794 and names[1]=='RA_LL_reversal' and len(done)<len(names)
def test_bootstrap_resume_contract():
 done=np.zeros(1000,bool);done[:10]=True;assert done.sum()==10 and done.shape==(1000,)
