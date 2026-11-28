"""Tests for frontier Pseudo-Voigt L1-NNLS deconvolution, MS/MS spectral entropy, and CIF/DIF/ROD parsers."""

from __future__ import annotations

import math
import random

import pytest

from lab_attrib import (
    AttributionOutcome,
    MassSpecRecord,
    build_gaussian_spectrum,
    build_pseudo_voigt_spectrum,
    caglioti_fwhm,
    convolve_pseudo_voigt_broadening,
    deconvolve_frontier_spectrum,
    deconvolve_multiphase_library,
    deconvolve_spectrum,
    detect_msms_chimera,
    diagnose_assay,
    elastic_nnls,
    evaluate_msms_orthogonal_readout,
    load_cif_diffractogram,
    load_raman_spectrum,
    load_xy_diffractogram,
    msms_cosine_similarity,
    msms_entropy_similarity,
    pseudo_voigt_profile,
    raman_to_jcamp_dx,
    spectral_entropy,
)


def test_elastic_nnls_numpy_stdlib_parity_kkt_and_l1_sparsity() -> None:
    pytest.importorskip("numpy")
    grid = [15.0 + 0.15 * i for i in range(240)]
    cols = [
        build_pseudo_voigt_spectrum(grid, [(22.0, 1.0, 0.45, 0.5), (38.0, 0.6, 0.45, 0.5)]),
        build_pseudo_voigt_spectrum(grid, [(29.4, 1.0, 0.45, 0.5), (43.2, 0.4, 0.45, 0.5)]),
        build_pseudo_voigt_spectrum(grid, [(26.6, 1.0, 0.45, 0.5), (50.1, 0.3, 0.45, 0.5)]),
        build_pseudo_voigt_spectrum(grid, [(33.1, 1.0, 0.45, 0.5)]),
        build_pseudo_voigt_spectrum(grid, [(47.5, 1.0, 0.45, 0.5)]),
    ]
    rng = random.Random(2026)
    obs = [
        max(0.0, 0.85 * cols[1][i] + 0.45 * cols[2][i] + rng.gauss(0.0, 0.015))
        for i in range(len(grid))
    ]
    weights = [1.0 + 0.5 * (i % 3) for i in range(len(grid))]

    res_np = elastic_nnls(
        cols,
        obs,
        l1_alpha=0.18,
        l2_alpha=1e-3,
        sample_weights=weights,
        use_numpy=True,
    )
    res_py = elastic_nnls(
        cols,
        obs,
        l1_alpha=0.18,
        l2_alpha=1e-3,
        sample_weights=weights,
        use_numpy=False,
    )
    assert res_np.converged and res_py.converged
    assert res_np.kkt_max_violation <= 1e-10
    assert res_py.kkt_max_violation <= 1e-10
    assert res_np.objective == pytest.approx(res_py.objective, abs=1e-9)
    for w_np, w_py in zip(res_np.weights, res_py.weights):
        assert w_np == pytest.approx(w_py, abs=1e-8)

    # Absent distractor phases (0, 3, 4) are driven strictly to 0.0 by L1 sparsity
    assert res_py.weights[0] == 0.0
    assert res_py.weights[3] == 0.0
    assert res_py.weights[4] == 0.0
    assert set(res_py.support_indices) == {1, 2}

    # Support debiasing eliminates L1 shrinkage bias on active phases 1 and 2
    res_debiased = elastic_nnls(
        cols,
        obs,
        l1_alpha=0.18,
        l2_alpha=1e-6,
        sample_weights=weights,
        debias_support=True,
        use_numpy=False,
    )
    assert res_debiased.weights[1] == pytest.approx(0.85, abs=0.025)
    assert res_debiased.weights[2] == pytest.approx(0.45, abs=0.025)
    assert res_debiased.weights[0] == 0.0


def test_shift_voigt_l1_nnls_beats_naive_and_rigid_nnls_under_shift_and_broadening() -> None:
    grid = [15.0 + 0.08 * i for i in range(625)]  # 15..65 deg
    # Sharp reference library (Gaussian FWHM = 0.22 deg) of 8 candidate minerals
    library_specs = [
        ("Calcite", [(23.05, 0.12), (29.40, 1.00), (35.97, 0.14), (39.41, 0.18), (43.16, 0.18)]),
        ("Aragonite", [(26.21, 1.00), (27.22, 0.55), (33.13, 0.46), (37.88, 0.38), (45.85, 0.52)]),
        ("Quartz", [(20.86, 0.22), (26.64, 1.00), (36.54, 0.08), (50.14, 0.14)]),
        ("Anatase", [(25.28, 1.00), (37.80, 0.20), (48.05, 0.35), (53.89, 0.20)]),
        ("Rutile", [(27.45, 1.00), (36.09, 0.48), (41.23, 0.22), (54.32, 0.58)]),
        ("Halite", [(31.69, 1.00), (45.45, 0.55), (56.48, 0.18)]),
        ("Witherite", [(23.90, 1.00), (24.30, 0.48), (34.09, 0.24), (41.98, 0.26)]),
        ("Fluorite", [(28.27, 1.00), (47.01, 0.55), (55.77, 0.30)]),
    ]
    sigma_sharp = 0.22 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    library = [
        (
            name,
            build_gaussian_spectrum(grid, [(ang, h, sigma_sharp) for ang, h in pks]),
        )
        for name, pks in library_specs
    ]

    # Experimental mixture of ONLY Calcite + Quartz with:
    #   +0.26 deg specimen-displacement zero-point shift,
    #   Pseudo-Voigt size-broadened FWHM = 0.52 deg (eta = 0.60 Lorentzian wings),
    #   smooth amorphous glass background halo + realistic detector noise!
    rng = random.Random(77)
    true_shift = 0.26
    cal_exp = build_pseudo_voigt_spectrum(
        grid,
        [(ang + true_shift, h, 0.52, 0.60) for ang, h in library_specs[0][1]],
    )
    qtz_exp = build_pseudo_voigt_spectrum(
        grid,
        [(ang + true_shift, h, 0.52, 0.60) for ang, h in library_specs[2][1]],
    )
    n = len(grid)
    observed = [
        max(
            0.0,
            0.85 * cal_exp[i]
            + 0.60 * qtz_exp[i]
            + 0.06 * math.sin(math.pi * i / float(n - 1))
            + rng.gauss(0.0, 0.012),
        )
        for i in range(n)
    ]

    naive_res = deconvolve_multiphase_library(observed, grid, library, method="naive_peak_cosine")
    rigid_res = deconvolve_multiphase_library(observed, grid, library, method="rigid_gaussian_nnls")
    frontier_res = deconvolve_multiphase_library(
        observed,
        grid,
        library,
        method="shift_voigt_l1_nnls",
        max_zero_shift=0.35,
    )

    true_set = {"Calcite", "Quartz"}
    assert set(frontier_res.detected_phases) == true_set
    assert set(naive_res.detected_phases) != true_set
    assert naive_res.rwp_permille > frontier_res.rwp_permille
    assert frontier_res.rwp_permille < rigid_res.rwp_permille
    assert frontier_res.rwp_permille <= 100
    assert rigid_res.rwp_permille > 350
    assert frontier_res.fitted_zero_shift == pytest.approx(true_shift, abs=0.04)

    # Also verify deconvolve_spectrum(shift_search_mode="golden_section") + Caglioti helpers
    assert caglioti_fwhm(40.0) > 0.20
    pv_single = pseudo_voigt_profile(grid, grid[188], 1.0, 0.4, eta=0.5)
    assert max(pv_single) == pytest.approx(1.0, abs=1e-9)
    conv = convolve_pseudo_voigt_broadening(library[0][1], grid, fwhm_add=0.30, eta=0.5)
    assert len(conv) == len(grid)

    gs_deconv = deconvolve_spectrum(
        observed,
        [library[2][1]],  # Quartz as precursor
        library[0][1],    # Calcite as target
        two_theta_grid=grid,
        max_peak_shift_deg=0.35,
        background_poly_degree=3,
        shift_search_mode="golden_section",
    )
    assert gs_deconv.target_signal_present is True
    assert gs_deconv.spectral_residual_permille <= 100
    assert diagnose_assay(gs_deconv.to_assay_telemetry()).outcome is AttributionOutcome.CERTIFIED_POSITIVE

    # And ensure deconvolve_frontier_spectrum catches shifted ghost precursor
    ghost_prec = build_pseudo_voigt_spectrum(
        grid, [(29.40 + 0.22, 1.0, 0.48, 0.55), (39.41 + 0.22, 0.18, 0.48, 0.55)]
    )
    ghost_res = deconvolve_frontier_spectrum(
        ghost_prec,
        [library[0][1]],
        library[0][1],
        two_theta_grid=grid,
        max_peak_shift_deg=0.35,
    )
    assert ghost_res.precursor_overlap_permille >= 400


def test_msms_spectral_entropy_similarity_and_chimera_quarantine() -> None:
    # Low-entropy spectrum dominated by molecular ion at m/z 250.10 plus 4 diagnostic minor fragments
    ref_ce20 = MassSpecRecord(
        identifier="REF_CE20",
        mzs=(105.034, 133.029, 161.060, 189.055, 250.108),
        intensities=(0.04, 0.05, 0.06, 0.04, 1.00),
        precursor_mz=250.108,
        inchikey="ABCDEFGHIJKLMN-UHFFFAOYSA-N",
    )
    # Higher collision energy (CE40): precursor m/z 250 drops 20x while diagnostic fragments dominate!
    obs_ce40 = MassSpecRecord(
        identifier="OBS_CE40",
        mzs=(105.035, 133.030, 161.061, 189.056, 250.108),
        intensities=(0.75, 0.90, 1.00, 0.65, 0.05),
        precursor_mz=250.108,
        inchikey="ABCDEFGHIJKLMN-UHFFFAOYSA-N",
    )
    # Unrelated compound that happens to share intact precursor peak at m/z 250.108 (survival ion trap!)
    decoy_unfragmented = MassSpecRecord(
        identifier="DECOY_SURVIVAL",
        mzs=(91.054, 119.049, 250.108),
        intensities=(0.05, 0.04, 1.00),
        precursor_mz=250.108,
        inchikey="ZYXWVUTSRQPONM-UHFFFAOYSA-N",
    )

    raw_s = spectral_entropy(ref_ce20.intensities, weighted=False)
    weighted_s = spectral_entropy(ref_ce20.intensities, weighted=True)
    assert raw_s < 3.0
    assert weighted_s > raw_s  # Li et al. (2021) dynamic power boost increases diagnostic entropy

    # Exact identity & disjoint-support boundary axioms
    assert msms_entropy_similarity(ref_ce20, ref_ce20, weighted=True) == pytest.approx(1.0, abs=1e-9)
    disjoint = MassSpecRecord(identifier="DISJOINT", mzs=(77.039, 95.049), intensities=(1.0, 0.5))
    assert msms_entropy_similarity(ref_ce20, disjoint, weighted=False) == pytest.approx(0.0, abs=1e-9)
    assert msms_entropy_similarity(ref_ce20, disjoint, weighted=True) == pytest.approx(0.0, abs=1e-9)

    cos_true = msms_cosine_similarity(obs_ce40, ref_ce20)
    ent_true = msms_entropy_similarity(obs_ce40, ref_ce20, weighted=True)
    cos_decoy_vs_ref = msms_cosine_similarity(decoy_unfragmented, ref_ce20)
    ent_decoy_vs_ref = msms_entropy_similarity(decoy_unfragmented, ref_ce20, weighted=True)

    # Cosine fails cross-CE (cos_true < 0.25) AND scores survival-ion decoy > 0.98!
    # Weighted spectral entropy recovers cross-CE true match (> 0.65) while penalizing decoy!
    assert cos_true < 0.30
    assert ent_true > 0.65
    assert ent_decoy_vs_ref < cos_decoy_vs_ref

    # Near-isobaric co-eluting chimera within +-0.6 Da Q1 quadrupole isolation window
    isobar_interferent = MassSpecRecord(
        identifier="ISOBAR_INTERF",
        mzs=(118.065, 146.060, 174.091, 250.520),
        intensities=(0.80, 1.00, 0.70, 0.30),
        precursor_mz=250.520,
        inchikey="NNNNNNNNNNNNNN-UHFFFAOYSA-N",
    )
    chimera_obs = MassSpecRecord(
        identifier="CHIMERA_OBS",
        mzs=(105.034, 118.065, 133.029, 146.060, 161.060, 174.091, 250.108),
        intensities=(0.45, 0.75, 0.55, 0.95, 0.60, 0.65, 0.50),
        precursor_mz=250.108,
    )
    chim_res = detect_msms_chimera(
        chimera_obs,
        obs_ce40,
        candidate_pool=[isobar_interferent, disjoint],
        isolation_window_da=1.0,
    )
    assert chim_res.is_chimera is True
    assert chim_res.interferent_id == "ISOBAR_INTERF"
    assert chim_res.entropy_gain_over_single > 0.05

    orth_chim = evaluate_msms_orthogonal_readout(
        chimera_obs,
        obs_ce40,
        similarity_metric="weighted_entropy",
        min_confirm_similarity=0.55,
        candidate_pool=[isobar_interferent],
        isolation_window_da=1.0,
    )
    assert orth_chim.selectivity_clean is False
    assert orth_chim.target_confirmed is False

    # Issue #16: Leave-interferent-out chimera detection via Q1 dual-precursor & unexplained residual entropy!
    chim_obs_dual = MassSpecRecord(
        identifier="CHIMERA_HELD_OUT_OBS",
        mzs=(105.034, 118.065, 133.029, 146.060, 161.060, 174.091, 250.108, 250.520),
        intensities=(0.45, 0.75, 0.55, 0.95, 0.60, 0.65, 0.50, 0.35),
        precursor_mz=250.108,
    )
    chim_held_out = detect_msms_chimera(
        chim_obs_dual,
        obs_ce40,
        candidate_pool=[disjoint],  # isobar_interferent strictly held out of pool!
        isolation_window_da=1.0,
    )
    assert chim_held_out.is_chimera is True
    assert chim_held_out.dual_precursor_detected is True
    assert chim_held_out.unexplained_intensity_fraction > 0.45


def test_dif_cif_and_rod_parsers_and_jcamp_roundtrip() -> None:
    dif_sample = """      Calcite
      CELL PARAMETERS:   4.9793  4.9793 17.0244   90.000   90.000  120.000
      SPACE GROUP: R-3c
      X-RAY WAVELENGTH:     1.541838
               2-THETA      INTENSITY    D-SPACING   H   K   L
                23.12          8.11        3.8468    0   1   2
                29.49        100.00        3.0292    1   0   4
                39.53         20.12        2.2799    1   1   3
=======================================================================
"""
    grid = [20.0 + 0.1 * i for i in range(251)]
    dif_rec = load_xy_diffractogram(dif_sample, resample_grid=grid)
    assert dif_rec.sample_id == "Calcite"
    assert len(dif_rec.intensities) == len(grid)
    peak_ang = grid[max(range(len(grid)), key=lambda idx: dif_rec.intensities[idx])]
    assert peak_ang == pytest.approx(29.5, abs=0.15)

    # Cubic rocksalt NaCl CIF structure -> verify (200) peak near ~31.7 deg Cu K-alpha
    nacl_cif = """data_NaCl
_chemical_formula_structural NaCl
_cell_length_a 5.6402(2)
_cell_length_b 5.6402(2)
_cell_length_c 5.6402(2)
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'x, 1/2+y, 1/2+z'
'1/2+x, y, 1/2+z'
'1/2+x, 1/2+y, z'
loop_
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na 0.0 0.0 0.0 1.0
Cl 0.5 0.5 0.5 1.0
"""
    cif_grid = [25.0 + 0.05 * i for i in range(501)]
    nacl_rec = load_cif_diffractogram(nacl_cif, grid=cif_grid)
    nacl_peak = cif_grid[max(range(len(cif_grid)), key=lambda idx: nacl_rec.intensities[idx])]
    assert nacl_peak == pytest.approx(31.70, abs=0.12)

    # ROD CIF-Raman -> RamanSpectrumRecord -> JCAMP-DX roundtrip
    rod_sample = """#\\#CIF_2.0
data_1000438
_chemical_compound_source R050130
_chemical_name_mineral Calcite
loop_
_raman_spectrum.raman_shift
_raman_spectrum.intensity
155.0 220.5
281.2 550.0
712.4 180.0
1085.6 1000.0
"""
    rod_rec = load_raman_spectrum(rod_sample)
    assert rod_rec.sample_id == "R050130"
    assert rod_rec.metadata["NAMES"] == "Calcite"
    jdx_text = raman_to_jcamp_dx(rod_rec)
    reloaded = load_raman_spectrum(jdx_text)
    assert reloaded.wavenumber_cm1 == rod_rec.wavenumber_cm1
    assert list(reloaded.intensities) == pytest.approx(list(rod_rec.intensities), abs=1e-4)
