import os

# 画像が格納されているディレクトリ
target_dir = "examples/notebooks/png_outputs"

# 現在のファイル名(key)と、新しいファイル名(value)の対応表
rename_map = {
    # --- Baseline Conditions ---
    "001_T96__The_8_Directional_Derivative_Formulas__Meridional_Plane__GSM___Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_0.0_nT__BzIMF_-5.0_nT.png":
        "T96_Derivs_Meridional_GSM_Base.png",

    "003_T96__The_8_Directional_Derivative_Formulas__Magnetic_Equator__SM_z_0__computed_in_GSM___Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_0.0_nT__BzIMF_-5.0_nT.png":
        "T96_Derivs_Equator_SM_Base.png",

    "004_T96__The_8_Directional_Derivative_Formulas__SM_z_0.5__computed_in_GSM___baseline___Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_0.0_nT__BzIMF_-5.0_nT.png":
        "T96_Derivs_Z05_SM_Base.png",

    "002_T96__The_9_Directional_Derivative_Formulas__GSM_XZ_Y-3__Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_0.0_nT__BzIMF_-5.0_nT.png":
        "T96_Derivs_Yminus3_GSM_Base.png",

    # --- Storm Conditions ---
    "008_T96__The_8_Directional_Derivative_Formulas__Meridional_Plane__GSM___Pdyn_10.60_nPa__Dst_-150_nT__ByIMF_0.0_nT__BzIMF_-15.0_nT.png":
        "T96_Derivs_Meridional_GSM_Storm.png",

    "010_T96__The_8_Directional_Derivative_Formulas__Magnetic_Equator__SM_z_0__computed_in_GSM___Pdyn_10.60_nPa__Dst_-150_nT__ByIMF_0.0_nT__BzIMF_-15.0_nT.png":
        "T96_Derivs_Equator_SM_Storm.png",

    "011_T96__The_8_Directional_Derivative_Formulas__SM_z_0.5__computed_in_GSM___Pdyn_10.60_nPa__Dst_-150_nT__ByIMF_0.0_nT__BzIMF_-15.0_nT.png":
        "T96_Derivs_Z05_SM_Storm.png",

    "009_T96__The_8_Directional_Derivative_Formulas__GSM_XZ_Y-3__Pdyn_10.60_nPa__Dst_-150_nT__ByIMF_0.0_nT__BzIMF_-15.0_nT.png":
        "T96_Derivs_Yminus3_GSM_Storm.png",

# --- Baseline Conditions ---
    # 1. Equator (z=0)
    "005_FAC4_SM_z0_baseline__Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_0.0_nT__BzIMF_-5.0_nT.png":
        "T96_FAC_Decomp_Equator_SM_Base.png",
    
    # 2. Off-Equator (z=0.5)
    "007_T96__FAC-related_4_panels_in_Magnetic_Equatorial_Parallel_Plane__SM_z_0.5___baseline___SM_z0.5__Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_0.0_nT__BzIMF_-5.0_nT.png":
        "T96_FAC_Decomp_Z05_SM_Base.png",
    
    # 3. Off-Meridian (y=-3)
    "006_T96__FAC-related_4_panels__baseline_SW_params___GSM_XZ_Y-3__Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_0.0_nT__BzIMF_-5.0_nT.png":
        "T96_FAC_Decomp_Yminus3_GSM_Base.png",

    # --- Storm Conditions ---
    # 4. Equator (z=0)
    "012_FAC4_SM_z0__Pdyn_10.60_nPa__Dst_-150_nT__ByIMF_0.0_nT__BzIMF_-15.0_nT.png":
        "T96_FAC_Decomp_Equator_SM_Storm.png",
    
    # 5. Off-Equator (z=0.5)
    "014_T96__FAC-related_4_panels_in_Magnetic_Equatorial_Parallel_Plane__SM_z_0.5___SM_z0.5__Pdyn_10.60_nPa__Dst_-150_nT__ByIMF_0.0_nT__BzIMF_-15.0_nT.png":
        "T96_FAC_Decomp_Z05_SM_Storm.png",
    
    # 6. Off-Meridian (y=-3)
    "013_T96__FAC-related_4_panels__GSM_XZ_Y-3__Pdyn_10.60_nPa__Dst_-150_nT__ByIMF_0.0_nT__BzIMF_-15.0_nT.png":
        "T96_FAC_Decomp_Yminus3_GSM_Storm.png",
        # --- 1. Geometric Coefficients (Derivs) ---
    # Meridional Plane (GSM y=0)
    "001_T96__The_8_Directional_Derivative_Formulas__Meridional_Plane__GSM___Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_5.0_nT__BzIMF_-5.0_nT.png":
        "T96_Derivs_Meridional_GSM_By5.png",

    # Equatorial Plane (SM z=0)
    "003_T96__The_8_Directional_Derivative_Formulas__Magnetic_Equator__SM_z_0__computed_in_GSM___Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_5.0_nT__BzIMF_-5.0_nT.png":
        "T96_Derivs_Equator_SM_By5.png",

    # Off-Equator Plane (SM z=0.5)
    "004_T96__The_8_Directional_Derivative_Formulas__SM_z_0.5__computed_in_GSM___baseline___Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_5.0_nT__BzIMF_-5.0_nT.png":
        "T96_Derivs_Z05_SM_By5.png",

    # Off-Meridian Plane (GSM y=-3)
    "002_T96__The_9_Directional_Derivative_Formulas__GSM_XZ_Y-3__Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_5.0_nT__BzIMF_-5.0_nT.png":
        "T96_Derivs_Yminus3_GSM_By5.png",

    # --- 2. FAC Decomposition ---
    # Equatorial Plane (SM z=0)
    "005_FAC4_SM_z0_baseline__Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_5.0_nT__BzIMF_-5.0_nT.png":
        "T96_FAC_Decomp_Equator_SM_By5.png",

    # Off-Equator Plane (SM z=0.5)
    "007_T96__FAC-related_4_panels_in_Magnetic_Equatorial_Parallel_Plane__SM_z_0.5___baseline___SM_z0.5__Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_5.0_nT__BzIMF_-5.0_nT.png":
        "T96_FAC_Decomp_Z05_SM_By5.png",

    # Off-Meridian Plane (GSM y=-3)
    "006_T96__FAC-related_4_panels__baseline_SW_params___GSM_XZ_Y-3__Pdyn_6.02_nPa__Dst_-30_nT__ByIMF_5.0_nT__BzIMF_-5.0_nT.png":
        "T96_FAC_Decomp_Yminus3_GSM_By5.png"
}

def rename_images():
    # スクリプト実行場所からの相対パス、もしくは絶対パスを確認してください
    current_dir = os.getcwd()
    work_dir = os.path.join(current_dir, target_dir)
    
    print(f"Target Directory: {work_dir}")

    if not os.path.exists(work_dir):
        print(f"[Error] Directory not found: {work_dir}")
        return

    for old_name, new_name in rename_map.items():
        old_path = os.path.join(work_dir, old_name)
        new_path = os.path.join(work_dir, new_name)

        if os.path.exists(old_path):
            try:
                os.rename(old_path, new_path)
                print(f"[OK] Renamed: {old_name} \n -> {new_name}")
            except Exception as e:
                print(f"[Error] Could not rename {old_name}: {e}")
        else:
            # 既にリネーム済みかチェック
            if os.path.exists(new_path):
                print(f"[Skip] Already renamed: {new_name}")
            else:
                print(f"[Missing] File not found: {old_name}")

if __name__ == "__main__":
    rename_images()