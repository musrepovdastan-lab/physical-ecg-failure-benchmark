"""Transpose already-computed mask APs to resample × mask orientation."""
import shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
tasks = ROOT / "outputs/mask_semantics_recovery_v2/bootstrap_tasks"
backup = tasks / "original_orientation_backup"
backup.mkdir(parents=True, exist_ok=True)
changed = []
for path in sorted(tasks.glob("*.npz")):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    mask = arrays["mask_ap"]
    if mask.shape == (1000, 792):
        continue
    if mask.shape != (792, 1000):
        raise RuntimeError(f"unexpected mask_ap shape {mask.shape}: {path.name}")
    if not np.isfinite(mask).all():
        raise RuntimeError(f"non-finite mask scores: {path.name}")
    backup_path = backup / path.name
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    arrays["mask_ap"] = mask.T.copy()
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    changed.append(path.name)
print({"transposed_tasks": len(changed), "names": changed, "backup_dir": str(backup.relative_to(ROOT))})
