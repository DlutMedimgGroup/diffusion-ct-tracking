#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from glob import glob
import numpy as np
import pandas as pd


# =============== 你需要改的路径 ===============
JSON_DIR = "/home/mingrui/disk1/datasets/DeepLesion/DLT_Annotations"
CSV_OUT  = "/home/mingrui/disk1/processed_dataset/DeepLesion_1mm/lesion_radius.csv"
# ============================================


# ====== 单位假设（重要）======
# 你的 JSON 里 recist coordinate 是像素坐标，recist diameter 很可能也是“像素长度(px)”
# True: 直径(px) * spacing(mm/px) -> 直径(mm)
# False: 直径已经是 mm
ASSUME_DIAMETER_IN_PIXEL = False


def compute_radius_mm(diam_2, spacing_xyz, assume_px=True):
    """
    等效圆半径（mm）:
      r = (d_long_mm + d_short_mm) / 4

    diam_2: [d_long, d_short]（通常来自 "source recist diameter"）
    spacing_xyz: [sx, sy, sz]（来自 "source spacing"）
    assume_px: diam_2 是否为像素长度
    """
    if diam_2 is None or spacing_xyz is None:
        return np.nan, np.nan, np.nan

    if not (isinstance(diam_2, (list, tuple)) and len(diam_2) >= 2):
        return np.nan, np.nan, np.nan
    if not (isinstance(spacing_xyz, (list, tuple)) and len(spacing_xyz) >= 2):
        return np.nan, np.nan, np.nan

    d_long = float(diam_2[0])
    d_short = float(diam_2[1])
    sx = float(spacing_xyz[0])
    sy = float(spacing_xyz[1])

    if assume_px:
        d_long_mm = d_long * sx
        d_short_mm = d_short * sy
    else:
        d_long_mm = d_long
        d_short_mm = d_short

    r_mm = d_long_mm / 2.0
    return r_mm, d_long_mm, d_short_mm


def main():
    json_files = sorted(glob(os.path.join(JSON_DIR, "*.json")))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No json found under: {JSON_DIR}")

    rows = []
    seen_pairs = set()  # 去掉 A->B / B->A 重复（与你预处理一致）

    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]  # train/test/valid
        with open(jf, "r") as f:
            data = json.load(f)

        for pid, ct in data.items():
            pair_id = f"{base}_{pid}"

            s_name = ct.get("source", "")
            t_name = ct.get("target", "")

            # 用 center 做去重 key（与你预处理同逻辑）
            s_center = tuple(np.round(np.array(ct.get("source center", [np.nan]*3), dtype=float), 3))
            t_center = tuple(np.round(np.array(ct.get("target center", [np.nan]*3), dtype=float), 3))
            key = (s_name, t_name, s_center, t_center)
            rev_key = (t_name, s_name, t_center, s_center)
            if key in seen_pairs or rev_key in seen_pairs:
                continue
            seen_pairs.add(key)

            src_diam = ct.get("source recist diameter", None)
            tgt_diam = ct.get("target recist diameter", None)
            src_spacing = ct.get("source spacing", None)
            tgt_spacing = ct.get("target spacing", None)

            src_r, src_long_mm, src_short_mm = compute_radius_mm(
                src_diam, src_spacing, assume_px=ASSUME_DIAMETER_IN_PIXEL
            )
            tgt_r, tgt_long_mm, tgt_short_mm = compute_radius_mm(
                tgt_diam, tgt_spacing, assume_px=ASSUME_DIAMETER_IN_PIXEL
            )

            rows.append({
                "pair_id": pair_id,
                "source": s_name,
                "target": t_name,

                "source_radius_mm": src_r,
                "target_radius_mm": tgt_r,

                "source_long_mm": src_long_mm,
                "source_short_mm": src_short_mm,
                "target_long_mm": tgt_long_mm,
                "target_short_mm": tgt_short_mm,

                # 方便你回溯检查
                "source_recist_slice": ct.get("source recist slice", ""),
                "target_recist_slice": ct.get("target recist slice", ""),
            })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    df.to_csv(CSV_OUT, index=False)

    print(f"[OK] Saved: {CSV_OUT}")
    print(f"[INFO] rows={len(df)}")
    if len(df) > 0:
        nan_cnt = int(df["source_radius_mm"].isna().sum() + df["target_radius_mm"].isna().sum())
        print(f"[INFO] NaN radius count (source+target) = {nan_cnt}")

    # 可选：快速看一下分布（避免单位搞错）
    if len(df) > 0:
        print("[INFO] radius mm (source) quantiles:")
        print(df["source_radius_mm"].quantile([0.0, 0.25, 0.5, 0.75, 0.95, 1.0]))


if __name__ == "__main__":
    main()