#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-13T14:33:43.066Z
"""

import pandas as pd
import numpy as np
from scipy.optimize import brentq
import math
from sklearn.model_selection import train_test_split

hb = 6.626e-34 / (2 * np.pi)                 #ħ
ev = 1.602e-19                               #電子伏特

def find_bound_states(a, m, V0):
    #--------------------------------------------
    #    給定井寬 a、質量 m、井深 V0
    #    求出所有束縛能階，回傳：
    #        [("even" , k , E) , ("odd" , k , E) , ...]
    #    依能量由小排到大
    
    #    井內區域 |x| < a/2 , V = 0 → 波函數可簡化為sin或cos(取決於奇偶性)
    #    井外 V=V0，若 E < V0 → 必定為指數衰減 e ^ {-κx} , κ為井外衰減常數
    #    束縛態需解超越方程：
    #    even: ktan(k a/2) = κ
    #    odd : kcot(k a/2) = -κ
    #    其中 κ = sqrt(2m(V0-E))/ħ ，而 k = sqrt(2mE)/ħ
    #    同時因tanθ和cotθ有特定點無法計算，所以需要特別標記
    #--------------------------------------------
    half = a / 2
    k0   = np.sqrt(2 * m * V0) / hb          # 對應 E = V0 的最大 k
    states = []
    
# 定義奇偶態的方程式
    def f_even(k):
        if(not (0 < k < k0)):
            return np.nan
        u = k * half
        return k * np.tan(u) - np.sqrt(k0*k0 - k*k)

    def f_odd(k):
        if(not (0 < k < k0)):
            return np.nan
        u = k * half
        return k / np.tan(u) + np.sqrt(k0*k0 - k*k)

#二分法求根(和勘根定理差不多)
    def solve_root(func, kL, kR, tol=1e-10):
        fL, fR = func(kL), func(kR) 
        if(np.isnan(fL) or np.isnan(fR) or fL * fR > 0):# 異號才有根
            return None

        for _ in range(60):        #夾60次
            kM = 0.5 * (kL + kR)
            fM = func(kM)
            if abs(fM) < tol:
                return kM
            if fL * fM <= 0:
                kR, fR = kM, fM
            else:
                kL, fL = kM, fM
        return 0.5 * (kL + kR)

#根落在哪些區間？依 tan/cot 奇點自動切割
    eps = 1e-6
    u_max = k0 * half
    n_max = int(u_max / np.pi + 3)

#偶宇態
    # tan(u)的奇點在 u = π/2 + nπ(n∈整數)，此處 tan(u) → ±∞，方程式無法定義，
    # 因此求根時必須將奇點當成區間分隔點，避免跨越或直接取樣到奇點。
    asym = [0]
    for n in range(n_max):
        val = (np.pi/2) + n*np.pi
        if val < u_max:
            asym.append(val)
    asym.append(u_max)
    asym = sorted(set(asym))

    for uL, uR in zip(asym[:-1], asym[1:]):
        kL = (uL + eps) / half
        kR = (uR - eps) / half
        root = solve_root(f_even, kL, kR)
        if root is not None:
            E = hb*hb*root*root / (2*m)
            if 0 < E < V0:
                states.append(("even", root, E))

#奇宇態
    # cot(u)的奇點在 u = nπ(n∈整數)，因為sin(nπ)=0，cot(u) → ±∞，方程式無法定義
    # 因此求根時必須將奇點當成區間分隔點，避免跨越或直接取樣到奇點。
    asym = [0]
    for n in range(1, n_max+1):
        val = n*np.pi
        if val < u_max:
            asym.append(val)
    asym.append(u_max)
    asym = sorted(set(asym))

    for uL, uR in zip(asym[:-1], asym[1:]):
        kL = (uL + eps) / half
        kR = (uR - eps) / half
        root = solve_root(f_odd, kL, kR)
        if root is not None:
            E = hb*hb*root*root / (2*m)
            if 0 < E < V0:
                states.append(("odd", root, E))

# 能量由小到大排序
    states.sort(key=lambda t: t[2])
    return states

# 計算 ψ(x)：井內 (sin/cos) + 井外 (exponential)

def compute_wavefunction(a, m, V0, parity, k, E, num_points=501):
    #--------------------------------------------
    #    基本參數:
    #        a(井寬), m(粒子質量), V0(井外勢能高度), parity(波函數的奇偶性),
    #        k(井內波數), E(能階的能量), 
    #        num_point(在x ∈ [−3a, 3a]的區間裡，均勻選出501個x)
    #    回傳：
    #        xs, xp(x',無因次化後的xs),
    #        psi_raw(未標準化的ψ), psi_norm(標準化後的ψ)
    
    #    物理定義：
    #    -|x| ≤ a/2 (井內)：
    #         even: ψ = cos(kx)
    #         odd : ψ = sin(kx)
    #    -|x| > a/2 (井外)：
    #         ψ = ψ(a/2) * exp(-κ(|x| - a/2))
    #      使用|x|，不需區分左右，波函數自然對稱。
    #--------------------------------------------

    xs = np.linspace(-3*a, 3*a, num_points)
    dx = xs[1] - xs[0]

    half = a / 2
    k0   = np.sqrt(2*m*V0) / hb
    kappa = np.sqrt(k0*k0 - k*k)  # 井外衰減常數

    psi_raw = np.zeros_like(xs)

    # 井界 |x| = a/2 處的值，用來做井外的接續
    boundary_val = math.cos(k*half) if parity == "even" else math.sin(k*half)

    for i, x in enumerate(xs):
        ax = abs(x)
        if(ax <= half):# 井內
            
            if(parity == "even"):
                psi_raw[i] = math.cos(k*x)
            else:
                psi_raw[i] = math.sin(k*x)
        else:
            # 井外(指數衰減)
            psi_raw[i] = boundary_val * math.exp(-kappa * (ax - half))

    # -------- 標準化：使 ∫|ψ|² dx = 1 --------
    norm = math.sqrt(np.sum(psi_raw**2) * dx)
    psi_norm = psi_raw / norm if norm > 0 else psi_raw.copy()

    # -------- 無因次化座標：x' = kx --------
    xp = k * xs
    return xs, xp, psi_raw, psi_norm


# 主程式：掃 a, m, V0，將所有資料寫成 CSV
def main():
    rows = []
    num_points = 501
    # a, m, V0 分別取 1..20
    for a in range(1 , 21):                      #井寬(公尺)
        for m in range(1 , 21):                  #質量(公斤)
            for v0 in range(1 , 21):             #井高(eV)
                #---------------------------------------------
                #讓尺度符合微觀的規模
                #後續程式之相關變數依然用a , m , v0(因為方便)，但單位已經轉換
                a_nm = a*1e-9                    #奈米a
                m_e = m*1e-31                    #電子m
                v0_j = v0*1.602e-19 * 0.01        #電子伏特轉為焦耳
                                                 #由於原數值出現了216能階，所以*0.01壓縮
                #---------------------------------------------

                # 求能階
                states = find_bound_states(a_nm, m_e, v0_j)
                if not states:
                    continue

                # 逐能階建立 ψ(x)
                for n, (parity, k, E) in enumerate(states, start=1):

                    xs, xp, psi_raw, psi_norm = compute_wavefunction(
                        a_nm, m_e, v0_j,
                        parity, k, E,
                        num_points
                    )

                    # 展開成一筆一筆的資料列
                    for x, xp, pr, pn in zip(xs, xp, psi_raw, psi_norm):
                        rows.append([a_nm, m_e, v0_j, n, x, xp, pr, pn])

    df = pd.DataFrame(rows, columns=[
        "a", "m", "V0", "n", "x", "x\'", "psi_raw", "psi_norm"
    ])

    # 打亂所有資料，避免 ANN 學到順序
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    #數量
    n_test  = int(0.0002 * len(df))           #訓練不用太多筆
    n_train = int(0.02 * len(df))             #原資料超千萬筆，取2%，共472092筆
    
    test_df  = df.iloc[:n_test]
    train_df = df.iloc[n_test : n_test + n_train]
    
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    
    train_df.to_csv("data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)
    print("data rows:", len(train_df))
    print("test data rows :", len(test_df))


if __name__ == "__main__":
    main()