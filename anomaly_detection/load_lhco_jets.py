
import argparse
import h5py
import numpy as np
from typing import Dict, Tuple, Optional

def masked_mean(data: np.ndarray, mask: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute mean over 'axis' using a 0/1 mask that marks valid entries.
    data shape should broadcast with mask along 'axis'.
    Returns NaN-safe means (fills where sum(mask)==0 with 0.0).
    """
    # Expand mask to match data dims along 'axis'
    if mask.ndim < data.ndim:
        expand_axes = [1] * (data.ndim - mask.ndim)
        mask = mask[..., *expand_axes]
    valid = mask.astype(bool)
    # Sum only valid entries
    num = np.sum(np.where(valid, data, 0.0), axis=axis)
    den = np.sum(valid, axis=axis)
    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den > 0)
    return out.astype(np.float32)

def safe_get(f: h5py.File, path: str) -> Optional[h5py.Dataset]:
    return f[path] if path in f else None

def inspect_file(h5_path: str) -> None:
    print(f"\n=== Inspecting: {h5_path} ===")
    with h5py.File(h5_path, "r") as f:
        print("Attributes:")
        for k, v in f.attrs.items():
            print(f"  - {k}: {v}")
        print("\nHierarchy:")
        def _printer(name):
            print(name)
        f.visit(_printer)
        def _safe_shape(name):
            try:
                obj = f[name]
                if isinstance(obj, h5py.Dataset):
                    return obj.shape, obj.dtype
            except Exception:
                pass
            return None, None
        print("\nDataset shapes/dtypes:")
        for name in f:
            pass  # top-level only
        # Walk all datasets
        f.visititems(lambda name, obj: print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                     if isinstance(obj, h5py.Dataset) else None)

def load_event_arrays(h5_path: str) -> Dict[str, np.ndarray]:
    """Load available arrays for jet1/jet2 and global fields."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        # Required / common datasets
        for key in ["jet1/4mom", "jet2/4mom",
                    "jet1/coords", "jet2/coords",
                    "jet1/features", "jet2/features",
                    "jet1/mask", "jet2/mask",
                    "jet_coords", "jet_features",
                    "signal"]:
            if key in f:
                out[key.replace("/", "_")] = f[key][...]
    return out

def build_features(arrs: Dict[str, np.ndarray],
                   include_4mom: bool = True,
                   include_coords_mean: bool = True,
                   include_feat_mean: bool = True,
                   include_global: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build a fixed-length feature matrix X and label vector y (if present)."""
    pieces = []

    # Per-jet: 4-momentum (px, py, pz, E) -> shape (N, 4)
    if include_4mom:
        for jet in ["jet1", "jet2"]:
            key = f"{jet}_4mom"
            if key in arrs:
                x = arrs[key]
                if x.ndim != 2 or x.shape[1] != 4:
                    raise ValueError(f"{key} expected shape (N,4), got {x.shape}")
                pieces.append(x.astype(np.float32))

    # Per-jet: masked means over constituents for coords and features
    for jet in ["jet1", "jet2"]:
        mask_key = f"{jet}_mask"
        mask = arrs.get(mask_key, None)
        if mask is None:
            continue
        # coords: (N, C, 2) -> masked mean -> (N,2)
        if include_coords_mean:
            coords_key = f"{jet}_coords"
            if coords_key in arrs:
                coords = arrs[coords_key]
                if coords.ndim != 3 or coords.shape[2] != 2:
                    raise ValueError(f"{coords_key} expected shape (N,C,2), got {coords.shape}")
                coords_mean = masked_mean(coords, mask, axis=1)  # (N,2)
                pieces.append(coords_mean)
        # features: (N, C, F) -> masked mean -> (N,F)
        if include_feat_mean:
            feats_key = f"{jet}_features"
            if feats_key in arrs:
                feats = arrs[feats_key]
                if feats.ndim != 3:
                    raise ValueError(f"{feats_key} expected shape (N,C,F), got {feats.shape}")
                feats_mean = masked_mean(feats, mask, axis=1)  # (N,F)
                pieces.append(feats_mean)

    # Global per-event vectors if present (already fixed-length)
    if include_global:
        for key in ["jet_coords", "jet_features"]:
            if key in arrs:
                x = arrs[key]
                # Flatten any trailing dims beyond the first
                x = x.reshape(x.shape[0], -1).astype(np.float32)
                pieces.append(x)

    if not pieces:
        raise RuntimeError("No features were constructed. Check dataset keys and flags.")

    X = np.concatenate(pieces, axis=1)
    y = arrs.get("signal", None)
    if y is not None:
        y = y.astype(np.int64).reshape(-1)
    return X, y

def save_npz(X: np.ndarray, y: Optional[np.ndarray], out_path: str) -> None:
    if y is None:
        np.savez_compressed(out_path, X=X)
    else:
        np.savez_compressed(out_path, X=X, y=y)

def main():
    parser = argparse.ArgumentParser(description="Load LHCO-style jet HDF5 and build ML-ready features.")
    parser.add_argument("--h5", required=True, help="Path to bg_N100.h5 or sn_N100.h5")
    parser.add_argument("--inspect", action="store_true", help="Print file hierarchy and dataset shapes")
    parser.add_argument("--no-4mom", action="store_true", help="Exclude per-jet 4-momentum")
    parser.add_argument("--no-coords-mean", action="store_true", help="Exclude per-jet mean coords")
    parser.add_argument("--no-feat-mean", action="store_true", help="Exclude per-jet mean features")
    parser.add_argument("--no-global", action="store_true", help="Exclude global jet_coords / jet_features datasets")
    parser.add_argument("--out", default=None, help="If set, save X (and y if present) to NPZ at this path")
    args = parser.parse_args()

    if args.inspect:
        inspect_file(args.h5)

    arrs = load_event_arrays(args.h5)
    X, y = build_features(
        arrs,
        include_4mom=not args.no_4mom,
        include_coords_mean=not args.no_coords_mean,
        include_feat_mean=not args.no_feat_mean,
        include_global=not args.no_global,
    )

    print(f"Built features: X shape = {X.shape}, dtype = {X.dtype}")
    if y is not None:
        print(f"Labels y shape = {y.shape}, unique = {np.unique(y)}")
    else:
        print("No 'signal' labels found in file.")

    if args.out:
        save_npz(X, y, args.out)
        print(f"Saved to {args.out}")

    # Small preview
    np.set_printoptions(precision=4, suppress=True)
    print("\nPreview X[0:3, 0:10]:")
    print(X[:3, :10])
    if y is not None:
        print("Preview y[0:10]:", y[:10])

if __name__ == "__main__":
    main()
