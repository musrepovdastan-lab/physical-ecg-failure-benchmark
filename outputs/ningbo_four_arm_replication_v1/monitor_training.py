"""Record wall time and GPU peak memory for a live workflow process."""
import json,sys,time,ctypes
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/ningbo_four_arm_replication_v1'; pid=int(sys.argv[1]); first_start=sys.argv[2]
path=OUT/'TRAINING_TIMINGS.json'; events=[]; active=None
while True:
    try:
        st=json.loads((OUT/'progress_resume.json').read_text(encoding='utf-8'))
    except Exception: st={}
    key=(st.get('arm'),st.get('seed')) if st.get('pid')==pid else None
    if key and key!=active:
        if active:
            events[-1]['end_utc']=datetime.now(timezone.utc).isoformat()
            events[-1]['wall_seconds']=(datetime.fromisoformat(events[-1]['end_utc'])-datetime.fromisoformat(events[-1]['start_utc'])).total_seconds()
        start=first_start if not events else datetime.now(timezone.utc).isoformat()
        events.append({'arm':key[0],'seed':key[1],'start_utc':start,'start_source':'observed UNIT_START/progress monitor','peak_gpu_memory_bytes':0})
        active=key
    if active and st.get('pid')==pid and (st.get('arm'),st.get('seed'))==active:
        events[-1]['peak_gpu_memory_bytes']=max(events[-1]['peak_gpu_memory_bytes'],int(st.get('gpu_peak_allocated_bytes',0) or 0))
        events[-1]['last_status_utc']=st.get('updated_utc')
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'trainer_pid':pid,'events':events,'updated_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8');tmp.replace(path)
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=(ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32);k.OpenProcess.restype=ctypes.c_void_p;k.GetExitCodeProcess.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32));k.GetExitCodeProcess.restype=ctypes.c_int;k.CloseHandle.argtypes=(ctypes.c_void_p,)
    h=k.OpenProcess(0x1000,False,pid)
    if not h: alive=ctypes.get_last_error()==5
    else:
        try: code=ctypes.c_uint32(); alive=bool(k.GetExitCodeProcess(h,ctypes.byref(code))) and code.value==259
        finally: k.CloseHandle(h)
    if not alive:
        if active:
            events[-1]['end_utc']=datetime.now(timezone.utc).isoformat()
            events[-1]['wall_seconds']=(datetime.fromisoformat(events[-1]['end_utc'])-datetime.fromisoformat(events[-1]['start_utc'])).total_seconds()
        tmp.write_text(json.dumps({'trainer_pid':pid,'events':events,'updated_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8');tmp.replace(path);break
    time.sleep(.7)
