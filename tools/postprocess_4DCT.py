# postprocess_4DCT.py
# ------------------------------------------------------------
# 批量读取指定文件夹下所有 “Case*.xlsx” 的稀疏点位移结果，
# 对 refined 坐标进行鲁棒平滑后处理，并输出：
# 1) 每个 case 的后处理结果 xlsx（新增 refined_pp 与误差等列）
# 2) 一个汇总结果文件 summary.xlsx（每个 case 一行）
#
# 数据列（单位 mm）：
#   moving: Z, moving: Y, moving: X         初始坐标
#   refined: Z, refined: Y, refined: X       待后处理的结果坐标
#   fixed_GT: Z, fixed_GT: Y, fixed_GT: X    金标准坐标（用于评价）
#
# 依赖：
#   pip install pandas openpyxl scipy numpy
# ------------------------------------------------------------

import os
import glob
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# =========================
# 1) 参数区（按需修改）
# =========================
INPUT_DIR = r"/home/mingrui/disk1/projects/20260112_DiffusionCorr/projects/inference_results/AE_1mm_local_4/4DCT_1mm"
OUTPUT_DIR = r"/home/mingrui/disk1/projects/20260112_DiffusionCorr/projects/inference_results/AE_1mm_local_4/postprocessed_4DCT"
SUMMARY_XLSX = "summary.xlsx"

FILE_GLOB = "Case*.xlsx"              # 批量处理 Case 开头的 xlsx

# kNN 超参
K_NEIGHBORS = 12                      # 建议 10~20
DIST_WEIGHT_EPS = 1e-12

ANCHOR_W_THRESH = 2.0   # 0.5~0.8 都可试；越大表示“更少点被固定”
ANCHOR_HARD = True      # True=严格固定；False=软固定（可微小移动）

# 鲁棒权重（Tukey bisquare）
TUKEY_C = 8.0                      # 标准值，越大越“宽松”
MIN_WEIGHT = 1e-3

# 图拉普拉斯平滑强度
LAMBDA_SMOOTH = 0.04                  # 越大越平滑（建议 0.2~5 试一下）

# 可选：对极端坏点直接剔除（仅用于拟合权重，不改变点数）
HARD_OUTLIER_Z = 16.0                  # 残差超过 z 倍 MAD 认为是强离群点（可设 None 关闭）

# 保存时是否覆盖
OVERWRITE = True


# =========================
# 2) 列名（保持与你xlsx一致）
# =========================
COL_MOVING = ["moving: Z", "moving: Y", "moving: X"]
COL_REFINED = ["refined: Z", "refined: Y", "refined: X"]
COL_GT = ["fixed_GT: Z", "fixed_GT: Y", "fixed_GT: X"]

COL_REFINED_PP = ["refined_pp: Z", "refined_pp: Y", "refined_pp: X"]


# =========================
# 3) 工具函数
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def mad(x: np.ndarray, eps: float = 1e-12) -> float:
    """Median Absolute Deviation -> robust scale (not multiplied by 1.4826 here)."""
    x = np.asarray(x).reshape(-1)
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + eps)


def robust_tukey_weights(r: np.ndarray, c: float = TUKEY_C) -> np.ndarray:
    """
    Tukey bisquare weights based on robust z = r / (1.4826*MAD).
    w = (1 - z^2)^2 for |z|<1 else 0
    """
    r = np.asarray(r).reshape(-1)
    s = 1.4826 * mad(r)  # robust sigma
    z = r / (c * s + 1e-12)
    w = np.zeros_like(z, dtype=np.float64)
    mask = np.abs(z) < 1.0
    t = 1.0 - z[mask] ** 2
    w[mask] = t ** 2
    w = np.clip(w, MIN_WEIGHT, 1.0)
    return w

def ckdtree_query(tree, x, k: int):
    """
    兼容不同 SciPy 版本的 cKDTree.query 并行参数：
    - 新版：workers
    - 少数版：n_jobs
    - 老版：无并行参数
    """
    try:
        return tree.query(x, k=k, workers=-1)
    except TypeError:
        try:
            return tree.query(x, k=k, n_jobs=-1)
        except TypeError:
            return tree.query(x, k=k)

def compute_knn_prediction_u(
    xyz: np.ndarray,
    u: np.ndarray,
    k: int = K_NEIGHBORS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    基于坐标 xyz 的 kNN 预测 u：u_hat_i = Σ w_ij u_j / Σ w_ij
    w_ij 使用 exp(-d^2/(2h^2))，h 用中位数邻距估计。
    返回：u_hat (N,3), residual_norm (N,), h
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n = xyz.shape[0]
    if n <= 2:
        return u.copy(), np.zeros((n,), dtype=np.float64), 1.0

    k_eff = min(k + 1, n)  # +1 因为会包含自己
    tree = cKDTree(xyz)
    dists, idxs = ckdtree_query(tree, xyz, k_eff)

    # 估计带宽 h：取每个点的第2近邻距离（排除自身0距离），再取中位数
    # dists[:, 0] ~ 0
    nn2 = dists[:, 1] if k_eff > 1 else np.full((n,), 1.0)
    h = float(np.median(nn2) + 1e-12)

    u_hat = np.zeros_like(u)
    for i in range(n):
        neigh = idxs[i]
        dist = dists[i]

        # 去掉自己（通常是第0个）
        mask = neigh != i
        neigh = neigh[mask]
        dist = dist[mask]
        if neigh.size == 0:
            u_hat[i] = u[i]
            continue

        # 距离权重
        w = np.exp(-(dist ** 2) / (2.0 * (h ** 2) + 1e-12))
        w_sum = float(np.sum(w) + DIST_WEIGHT_EPS)
        u_hat[i] = (w[:, None] * u[neigh]).sum(axis=0) / w_sum

    residual = np.linalg.norm(u - u_hat, axis=1)
    return u_hat, residual, h


def graph_laplacian_smooth_anchor(
    xyz: np.ndarray,
    u: np.ndarray,
    w_data: np.ndarray,
    k: int,
    h: float,
    lam: float,
    w_anchor: float = 0.6,
    anchor_hard: bool = True,
) -> np.ndarray:
    """
    只平滑“坏点”(w<w_anchor)，把“好点”(w>=w_anchor)当作 anchor。
    - hard anchor: 好点完全不动 u'_i = u_i
    - soft anchor: 好点给予极大 data weight，使其几乎不动（仍可微小调整）

    目标（对坏点集合 B 求解）：
        min_{u'_B}  Σ_{i∈B} w_i||u'_i - u_i||^2
                   + lam Σ_{(i,j)∈E} a_ij ||u'_i - u'_j||^2
    其中若 j 是 anchor，则 u'_j 视为常量 u_j。
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    w_data = np.asarray(w_data, dtype=np.float64).reshape(-1)

    n = xyz.shape[0]
    if n <= 2:
        return u.copy()

    # anchor / bad mask
    anchor_mask = w_data >= float(w_anchor)
    bad_mask = ~anchor_mask
    print(f"anchors={anchor_mask.mean()*100:.1f}% bad={bad_mask.mean()*100:.1f}%")

    # 若全是 anchor 或全是 bad
    if bad_mask.sum() == 0:
        return u.copy()

    # 1) 建图：kNN 邻接 A
    k_eff = min(k + 1, n)
    tree = cKDTree(xyz)
    dists, idxs = ckdtree_query(tree, xyz, k_eff)

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        neigh = idxs[i]
        dist = dists[i]
        mask = neigh != i
        neigh = neigh[mask]
        dist = dist[mask]
        if neigh.size == 0:
            continue
        a = np.exp(-(dist ** 2) / (2.0 * (h ** 2) + 1e-12))
        for j, aij in zip(neigh, a):
            if aij > A[i, j]:
                A[i, j] = aij
                A[j, i] = aij

    D = np.diag(A.sum(axis=1))
    L = D - A  # (N,N)

    # 2) 组装线性系统，只对坏点子集 B 解
    bad_idx = np.where(bad_mask)[0]
    anc_idx = np.where(anchor_mask)[0]

    # 子矩阵
    L_BB = L[np.ix_(bad_idx, bad_idx)]     # |B|x|B|
    L_BA = L[np.ix_(bad_idx, anc_idx)]     # |B|x|A|

    # data term 权重（只对坏点）
    Wb = np.diag(np.clip(w_data[bad_idx], MIN_WEIGHT, 1.0))

    # hard anchor：anchor 不出现在未知量里
    # 系统： (Wb + lam*L_BB) u'_B = Wb u_B - lam * L_BA u_A
    Mb = Wb + lam * L_BB

    rhs = (Wb @ u[bad_idx]) - lam * (L_BA @ u[anc_idx])

    # 3) 解三个分量
    u_out = u.copy()
    for c in range(3):
        u_out[bad_idx, c] = np.linalg.solve(Mb, rhs[:, c])

    if not anchor_hard:
        # soft anchor：给 anchor 极大的数据权重，让它“几乎不动”
        # 这里提供一个简单做法：再做一次全量解，但 anchor 权重极大
        big = 1e6
        W = np.diag(np.where(anchor_mask, big, np.clip(w_data, MIN_WEIGHT, 1.0)))
        M = W + lam * L
        rhs2 = W @ u
        for c in range(3):
            u_out[:, c] = np.linalg.solve(M, rhs2[:, c])

    return u_out


def tre_stats(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err).reshape(-1)
    if err.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(err)),
        "std": float(np.std(err, ddof=0)),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(np.max(err)),
    }


# =========================
# 4) 主流程（单 case）
# =========================
@dataclass
class CaseResult:
    case_name: str
    n_points: int
    tre_before_mean: float
    tre_before_std: float 
    tre_before_median: float
    tre_before_p90: float
    tre_before_max: float
    tre_after_mean: float
    tre_after_std: float
    tre_after_median: float
    tre_after_p90: float
    tre_after_max: float
    outlier_rate_wlt02: float
    h_nn_median: float


def process_one_xlsx(path_xlsx: str, out_dir: str) -> CaseResult:
    df = pd.read_excel(path_xlsx, engine="openpyxl")

    # 检查列
    for c in (COL_MOVING + COL_REFINED + COL_GT):
        if c not in df.columns:
            raise KeyError(f"Missing column: {c} in {path_xlsx}")

    moving = df[COL_MOVING].to_numpy(dtype=np.float64)   # (N,3) [Z,Y,X]
    refined = df[COL_REFINED].to_numpy(dtype=np.float64) # (N,3)
    gt = df[COL_GT].to_numpy(dtype=np.float64)           # (N,3)

    u = refined - moving  # displacement (N,3)

    # (1) kNN 预测 + 残差
    u_hat, r, h = compute_knn_prediction_u(moving, u, k=K_NEIGHBORS)

    # (2) 鲁棒权重
    w = robust_tukey_weights(r, c=TUKEY_C)

    # 可选硬剔除：对特别离群点权重进一步压小
    if HARD_OUTLIER_Z is not None:
        s = 1.4826 * mad(r)
        z = r / (s + 1e-12)
        w[z > float(HARD_OUTLIER_Z)] = MIN_WEIGHT

    # (3) 图拉普拉斯平滑（仅在点上）
    u_s = graph_laplacian_smooth_anchor(
        xyz=moving,
        u=u,
        w_data=w,
        k=K_NEIGHBORS,
        h=h,
        lam=LAMBDA_SMOOTH,
        w_anchor=ANCHOR_W_THRESH,
        anchor_hard=ANCHOR_HARD,
    )

    refined_pp = moving + u_s

    # (4) 评价 TRE（mm）
    tre_before = np.linalg.norm(refined - gt, axis=1)
    tre_after = np.linalg.norm(refined_pp - gt, axis=1)

    st_b = tre_stats(tre_before)
    st_a = tre_stats(tre_after)

    # (5) 写回每点结果
    df[COL_REFINED_PP[0]] = refined_pp[:, 0]
    df[COL_REFINED_PP[1]] = refined_pp[:, 1]
    df[COL_REFINED_PP[2]] = refined_pp[:, 2]

    df["u: dZ (mm)"] = u[:, 0]
    df["u: dY (mm)"] = u[:, 1]
    df["u: dX (mm)"] = u[:, 2]

    df["u_pp: dZ (mm)"] = u_s[:, 0]
    df["u_pp: dY (mm)"] = u_s[:, 1]
    df["u_pp: dX (mm)"] = u_s[:, 2]

    df["knn_residual (mm)"] = r
    df["robust_weight"] = w
    df["TRE_before (mm)"] = tre_before
    df["TRE_after (mm)"] = tre_after

    case_name = os.path.splitext(os.path.basename(path_xlsx))[0]
    out_path = os.path.join(out_dir, f"{case_name}_post.xlsx")
    if (not OVERWRITE) and os.path.exists(out_path):
        raise FileExistsError(f"Output exists: {out_path}")
    df.to_excel(out_path, index=False, engine="openpyxl")

    outlier_rate = float(np.mean(w < 0.2))

    return CaseResult(
        case_name=case_name,
        n_points=int(df.shape[0]),
        tre_before_mean=st_b["mean"],
        tre_before_std=st_b["std"],
        tre_before_median=st_b["median"],
        tre_before_p90=st_b["p90"],
        tre_before_max=st_b["max"],
        tre_after_mean=st_a["mean"],
        tre_after_std=st_a["std"],
        tre_after_median=st_a["median"],
        tre_after_p90=st_a["p90"],
        tre_after_max=st_a["max"],
        outlier_rate_wlt02=outlier_rate,
        h_nn_median=float(h),
    ), tre_before, tre_after


# =========================
# 5) 批处理入口
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    # --------- 用于全体 landmark 统计 ---------
    all_tre_before = []
    all_tre_after = []

    pattern = os.path.join(INPUT_DIR, FILE_GLOB)
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {pattern}")

    results: List[CaseResult] = []

    for f in files:
        try:
            # 注意：process_one_xlsx 返回 3 个值：res, tre_before, tre_after
            res, tre_b, tre_a = process_one_xlsx(f, OUTPUT_DIR)
            results.append(res)

            # 收集点级 TRE
            all_tre_before.append(tre_b)
            all_tre_after.append(tre_a)

            print(
                f"[OK] {res.case_name}: "
                f"TRE mean±std {res.tre_before_mean:.3f}±{res.tre_before_std:.3f} -> "
                f"{res.tre_after_mean:.3f}±{res.tre_after_std:.3f} mm | "
                f"p90 {res.tre_before_p90:.3f} -> {res.tre_after_p90:.3f} mm | "
                f"outlier(w<0.2)={res.outlier_rate_wlt02*100:.1f}%"
            )

        except Exception as e:
            print(f"[FAIL] {os.path.basename(f)}: {repr(e)}")

    # --------- 汇总 case-level 结果 ---------
    if len(results) == 0:
        print("\nNo successful cases. Summary not generated.")
        return

    sum_df = pd.DataFrame([r.__dict__ for r in results])
    sum_df = sum_df.sort_values(by="tre_after_mean", ascending=True)

    # --------- 计算全体 landmark 级别统计 ---------
    tre_b_all = np.concatenate(all_tre_before)
    tre_a_all = np.concatenate(all_tre_after)

    b_mean = float(np.mean(tre_b_all))
    b_std  = float(np.std(tre_b_all, ddof=0))
    b_med  = float(np.median(tre_b_all))
    b_p90  = float(np.percentile(tre_b_all, 90))
    b_max  = float(np.max(tre_b_all))

    a_mean = float(np.mean(tre_a_all))
    a_std  = float(np.std(tre_a_all, ddof=0))
    a_med  = float(np.median(tre_a_all))
    a_p90  = float(np.percentile(tre_a_all, 90))
    a_max  = float(np.max(tre_a_all))

    print("\n========== GLOBAL TRE over ALL landmarks ==========")
    print(f"Before: mean±std={b_mean:.4f}±{b_std:.4f}  median={b_med:.4f}  p90={b_p90:.4f}")
    print(f"After : mean±std={a_mean:.4f}±{a_std:.4f}  median={a_med:.4f}  p90={a_p90:.4f}")

    # --------- 把 global 统计写入 summary.xlsx 最后一行 ---------
    global_row = {
        "case_name": "__GLOBAL_ALL_POINTS__",
        "n_points": int(tre_b_all.size),

        "tre_before_mean": b_mean,
        "tre_before_std": b_std,
        "tre_before_median": b_med,
        "tre_before_p90": b_p90,
        "tre_before_max": b_max,

        "tre_after_mean": a_mean,
        "tre_after_std": a_std,
        "tre_after_median": a_med,
        "tre_after_p90": a_p90,
        "tre_after_max": a_max,

        "outlier_rate_wlt02": np.nan,
        "h_nn_median": np.nan,
    }

    sum_df = pd.concat([sum_df, pd.DataFrame([global_row])], ignore_index=True)

    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_XLSX)
    sum_df.to_excel(summary_path, index=False, engine="openpyxl")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()