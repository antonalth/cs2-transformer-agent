#!/usr/bin/env python3
"""
Proof‑of‑concept: extract *raw* vs. *recoil‑corrected* view‑angle deltas for the
first **N** ticks of a CS‑2 demo.

* For every player we detect eye‑angle and view‑punch fields automatically.
* We print the first *N* ticks (default 100) containing:
      tick  steamid  raw_dyaw  raw_dpitch  corr_dyaw  corr_dpitch
  where
      raw_d*     = current eyeAngle − previous eyeAngle
      corr_d*    = (eyeAngle − punch) − (prev eyeAngle − prev punch)

Later this table can be streamed into a DB per your pipeline; for now it just
prints to stdout for inspection.

Example
-------
    python recoil_poc.py demo.dem --rows 150 --debug
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
from demoparser2 import DemoParser

# ───────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────

def _xy(val):
    """Return (pitch,yaw) floats regardless of format."""
    if val is None:
        return 0.0, 0.0
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return float(val[0]), float(val[1])
    if isinstance(val, dict):
        return float(val.get('x', 0.0)), float(val.get('y', 0.0))
    if isinstance(val, str):
        try:
            obj = json.loads(val)
            if isinstance(obj, dict):
                return float(obj.get('x', 0.0)), float(obj.get('y', 0.0))
        except Exception:
            pass
        parts = [float(x) for x in val.replace(',', ' ').split() if x.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
    return 0.0, 0.0

# ───────────────────────────────────────────────────────
# field detection
# ───────────────────────────────────────────────────────
EYE_CANDIDATES = [
    'CCSPlayerPawn.m_angEyeAngles',
    'CCSPlayerPawn.CCSPlayerPawn.m_angEyeAngles',
    'CCSPlayerPawn.m_qDeathEyeAngles',
]
PUNCH_CANDIDATES = [
    'CCSPlayerPawn.CCSPlayer_CameraServices.m_vecCsViewPunchAngle',
    'CCSPlayerPawn.m_vecCsViewPunchAngle',
    'CCSPlayerPawn.m_aimPunchAngle',
]

# ───────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────

def cli():
    p = argparse.ArgumentParser(description="POC: compare raw vs recoil‑corrected angle deltas")
    p.add_argument('demofile', type=Path)
    p.add_argument('--rows', type=int, default=100, help='Rows to print (default 100)')
    p.add_argument('--debug', action='store_true')
    return p

def load_df(demo: Path, debug: bool):
    dp = DemoParser(demo.as_posix())
    wanted = ['tick', 'steamid', 'name'] + EYE_CANDIDATES + PUNCH_CANDIDATES
    df = pd.DataFrame(dp.parse_ticks(wanted_props=wanted))

    # eye
    eye_field = next((c for c in EYE_CANDIDATES if c in df.columns), None)
    if eye_field is None:
        raise RuntimeError('No eye‑angle field in demo')
    df[['eye_pitch','eye_yaw']] = df[eye_field].apply(lambda v: pd.Series(_xy(v)))

    # punch (optional)
    punch_field = next((c for c in PUNCH_CANDIDATES if c in df.columns), None)
    if punch_field:
        df[['punch_pitch','punch_yaw']] = df[punch_field].apply(lambda v: pd.Series(_xy(v)))
    else:
        if debug:
            print('[WARN] no punch‑angle field found -> assuming zero recoil')
        df['punch_pitch'] = 0.0; df['punch_yaw'] = 0.0

    return df[['tick','steamid','name','eye_pitch','eye_yaw','punch_pitch','punch_yaw']]


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add raw and recoil-corrected deltas (per player)."""
    # Ensure sorted for grouping
    df.sort_values(['steamid', 'tick'], inplace=True, ignore_index=True)

    # Raw diffs on eye angles
    df[['raw_dpitch', 'raw_dyaw']] = df.groupby('steamid')[['eye_pitch', 'eye_yaw']].diff()

    # Compute corrected eye angles (eye minus punch)
    df['corr_eye_pitch'] = df['eye_pitch'] - df['punch_pitch']
    df['corr_eye_yaw'] = df['eye_yaw'] - df['punch_yaw']

    # Corrected diffs on corrected eye angles
    df[['corr_dpitch', 'corr_dyaw']] = df.groupby('steamid')[['corr_eye_pitch', 'corr_eye_yaw']].diff()

    return df


def main():
    args = cli().parse_args()
    if not args.demofile.exists():
        sys.exit('Demo not found: ' + str(args.demofile))

    print('[parsing demo]')
    df = load_df(args.demofile, debug=args.debug)
    df = compute_deltas(df)

    # Replace NaN diffs (first tick per player) with zeros
    df[['raw_dpitch','raw_dyaw','corr_dpitch','corr_dyaw']] = \
        df[['raw_dpitch','raw_dyaw','corr_dpitch','corr_dyaw']].fillna(0)

    # Filter to ticks where recoil correction changed the delta
    df_diff = df[(df['raw_dyaw'] != df['corr_dyaw']) | (df['raw_dpitch'] != df['corr_dpitch'])]

    print(f"[showing first {args.rows} ticks with actual correction]")
    cols = ['tick','name','raw_dyaw','raw_dpitch','corr_dyaw','corr_dpitch']
    # If fewer rows than requested, show all
    to_show = df_diff.head(args.rows)
    print(to_show[cols].to_string(index=False))

if __name__ == '__main__':
    main()
