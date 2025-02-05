#![feature(f128)]
use ndarray::{Array2, ArrayView2};
use rand::Rng;
use std::{
    collections::{HashMap, HashSet},
    f64::{self, INFINITY},
};
use logaddexp::LogSumExp;
use num_traits::{Float,FloatConst};
use itertools::iproduct;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::{prelude::*, types::PyDict};

#[cfg(feature = "python")]
#[pymodule]
fn stochastic_greedy_model<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_probabilistic_gce, m)?)?;
    m.add_function(wrap_pyfunction!(py_window_nuc_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_window_nuc_rate_k, m)?)?;
    m.add_function(wrap_pyfunction!(py_window_score, m)?)?;
    m.add_function(wrap_pyfunction!(py_window_k_array_with_bonds, m)?)?;
    m.add_function(wrap_pyfunction!(py_window_path_nuc_rate, m)?)?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction(name = "probabilistic_gce")]
fn py_probabilistic_gce(
    py: Python<'_>,
    farray: PyReadonlyArray2<f64>,
    trials: usize,
    params: KTAMParams,
    depth: usize,
    calc_weightsize_g: bool,
    store_assemblies: bool,
) -> PyResult<PyObject> {
    let result = probabilistic_gce(
        farray.as_array(),
        trials,
        &params,
        depth,
        calc_weightsize_g,
        store_assemblies,
    );

    let dict = PyDict::new(py);
    dict.set_item("gce", result.gce)?;
    dict.set_item("gcns", result.gcns)?;
    dict.set_item("nucrate", result.nucrate)?;
    dict.set_item("nucrates", result.nucrates)?;
    dict.set_item("traces", result.traces)?;
    dict.set_item("size_traces", result.size_traces)?;
    dict.set_item("stopped_pct", result.stopped_pct)?;
    dict.set_item("ncns", result.ncns)?;
    dict.set_item("min_gcn_per_site", result.min_gcn_per_site.into_pyarray(py))?;
    dict.set_item(
        "weight_gcn_per_site",
        result.weight_gcn_per_site.into_pyarray(py),
    )?;
    dict.set_item(
        "min_size_per_site",
        result.min_size_per_site.into_pyarray(py),
    )?;
    dict.set_item(
        "weight_size_per_site",
        result.weight_size_per_site.into_pyarray(py),
    )?;
    dict.set_item("nucrate_per_site", result.nucrate_per_site.into_pyarray(py))?;
    dict.set_item("min_size", result.min_size)?;
    dict.set_item("weight_size", result.weight_size)?;
    dict.set_item("final_g", result.final_g)?;
    dict.set_item("base_g", result.base_g)?;
    dict.set_item("num_per_size", result.num_per_size)?;
    dict.set_item("g_weighted_per_size", result.g_weighted_per_size)?;

    Ok(dict.into())
}


#[cfg(feature = "python")]
#[pyfunction(name = "window_nuc_rate_k")]
fn py_window_nuc_rate_k(
    concarray: PyReadonlyArray2<f64>,
    gse: f64,
    alpha: f64,
    base: f64,
    kf: f64,
    k: usize
) -> PyResult<f64> {
    Ok(window_nuc_k(
        concarray.as_array(),
        k,
        gse,
        alpha,
        base,
        kf,
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "window_nuc_rate")]
fn py_window_nuc_rate(
    concarray: PyReadonlyArray2<f64>,
    gse: f64,
    alpha: f64,
    kf: f64,
    k_min: usize,
    k_max: usize,
) -> PyResult<f64> {
    Ok(window_nuc_rate(
        concarray.as_array(),
        gse,
        alpha,
        kf,
        k_min..k_max,
    ))
}

#[cfg(feature = "python")]
#[pyfunction(name = "window_score")]
fn py_window_score(_py: Python<'_>, z: PyReadonlyArray2<f64>, k: usize) -> PyResult<f64> {
    Ok(window_score(z.as_array(), k))
}

#[cfg(feature = "python")]
#[pyfunction(name = "window_k_array_with_bonds")]
fn py_window_k_array_with_bonds<'py>(
    py: Python<'py>,
    conc_array: PyReadonlyArray2<f64>,
    gse_to_e: PyReadonlyArray2<f64>,
    gse_to_s: PyReadonlyArray2<f64>,
    k: usize,
) -> Bound<'py, PyArray2<f64>> {
    let result = window_k_array_with_bonds(
        conc_array.as_array(),
        gse_to_e.as_array(),
        gse_to_s.as_array(),
        k,
    );
    result.into_pyarray(py)
}

#[cfg(feature = "python")]
#[pyfunction(name = "window_path_nuc_rate")]
fn py_window_path_nuc_rate(
    concarray: PyReadonlyArray2<f64>,
    gse: f64,
    alpha: f64,
    kf: f64,
    include_size_in_forward: bool,
) -> PyResult<f64> {
    Ok(window_path_nuc_rate(
        concarray.as_array(),
        gse,
        alpha,
        kf,
        include_size_in_forward,
    ))
}

pub const OFF4: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub const OFF5: [(i32, i32); 5] = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)];

#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub struct KTAMParams {
    gmc: f64,
    gse: f64,
    alpha: f64,
    kf: f64,
}

impl KTAMParams {
    pub fn new(gmc: f64, gse: f64, alpha: f64, kf: f64) -> Self {
        Self {
            gmc,
            gse,
            alpha,
            kf,
        }
    }

    pub fn g_se(&self) -> f64 {
        self.gse
    }
    pub fn g_mc(&self) -> f64 {
        self.gmc
    }
    pub fn k_f(&self) -> f64 {
        self.kf
    }
    pub fn epsilon(&self) -> f64 {
        2.0 * self.gse - self.gmc
    }
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
    pub fn kh_f(&self) -> f64 {
        self.kf * (-self.alpha).exp()
    }
}

/// Result of a single trajectory calculation
#[derive(Clone)]
pub struct TrajectoryResult {
    pub gcn: f64,
    pub cn: Array2<bool>,
    pub g_trace: Vec<f64>,
    pub crit_step: usize,
    pub size_trace: Vec<i64>,
    pub nucrate: f64,
    pub assemblies: Vec<Array2<bool>>,
}

// Helper function to choose position based on weights
fn choose_position(weights: ArrayView2<f64>) -> (usize, usize) {
    let mut rng = rand::thread_rng();
    let total = weights.sum();
    let trigger = total * rng.gen::<f64>();
    let mut accum = 0.0;

    for (idx, &weight) in weights.indexed_iter() {
        accum += weight;
        if accum >= trigger {
            return idx;
        }
    }

    // Fallback to last position if we somehow don't find one
    let shape = weights.shape();
    (shape[0] - 1, shape[1] - 1)
}

// Core functions for calculating properties
pub fn calculate_n(state: ArrayView2<bool>) -> usize {
    state.iter().filter(|&&x| x).count()
}

pub fn calculate_n_add(state: ArrayView2<bool>) -> usize {
    let mut n = 0;
    state.iter().for_each(|&x| n += x as usize);
    n
}

fn calculate_b(state: ArrayView2<bool>) -> usize {
    let (rows, cols) = state.dim();
    let mut bonds = 0;

    // Horizontal bonds
    for i in 0..rows {
        for j in 0..(cols - 1) {
            if state[[i, j]] && state[[i, j + 1]] {
                bonds += 1;
            }
        }
    }

    // Vertical bonds
    for i in 0..(rows - 1) {
        for j in 0..cols {
            if state[[i, j]] && state[[i + 1, j]] {
                bonds += 1;
            }
        }
    }

    bonds
}

fn calculate_g(conc_mult: ArrayView2<f64>, state: ArrayView2<bool>, params: &KTAMParams) -> f64 {
    let n = calculate_n(state) as f64;
    let b = calculate_b(state) as f64;
    let log_sum: f64 = state
        .iter()
        .zip(conc_mult.iter())
        .filter(|(&s, _)| s)
        .map(|(_, &c)| c.ln())
        .sum();

    n * params.g_mc() - b * params.g_se() - log_sum - params.alpha()
}

/// Calculate a single SGM trajectory
pub fn probable_trajectory(
    farray: ArrayView2<f64>,
    starting_site: (usize, usize),
    params: &KTAMParams,
    max_steps: usize,
    store_assemblies: bool,
) -> TrajectoryResult {
    let shape = farray.dim();
    let mut state = Array2::from_elem(shape, false);

    let mut current_cn = Array2::from_elem(shape, false);
    let mut current_gcn = f64::NEG_INFINITY;
    let mut current_critstep = 1;
    let mut current_frate = 0.0;

    let mut assemblies = Vec::new();
    let mut trace = Vec::new();
    let mut size_trace = Vec::new();

    // Stop if initial site has no tile
    if farray[[starting_site.0, starting_site.1]] == 0.0 {
        return TrajectoryResult {
            gcn: f64::INFINITY,
            cn: current_cn,
            g_trace: trace,
            crit_step: current_critstep,
            size_trace,
            nucrate: current_frate,
            assemblies,
        };
    }

    // Add initial tile to state
    state[[starting_site.0, starting_site.1]] = true;
    let mut statesize = 1;

    // Calculate initial G
    let mut g = params.g_mc() - params.alpha() - farray[[starting_site.0, starting_site.1]].ln();
    trace.push(g);
    size_trace.push(statesize as i64);

    let mut stepnum = 2;
    let mut dgatt = Array2::from_elem(shape, f64::INFINITY);
    let mut probs = Array2::zeros(shape);

    update_dgatt_and_probs_around(
        farray,
        &mut dgatt,
        &mut probs,
        state.view(),
        starting_site,
        params,
    );

    while stepnum <= max_steps {
        let (site, dg) =
            match probabilistic_step(farray, &mut dgatt, &mut probs, &mut state, params) {
                Some((s, d)) => (s, d),
                None => break, // No more steps possible
            };

        statesize += 1;

        let old_g = g;
        let pdg = dg;
        g += dg;
        trace.push(g);
        size_trace.push(statesize as i64);

        if store_assemblies {
            assemblies.push(state.clone());
        }

        // Check if we're in a state of higher G
        if g > current_gcn {
            current_gcn = g;
            current_critstep = stepnum;
            current_cn.assign(&state);

            // Calculate rates for critical nucleus
            let (dg_fill, frate, ntiles) = fill_favorable(
                farray, &mut dgatt, &mut probs, &mut state, site, params, true,
            );
            current_frate = frate * (-current_gcn).exp();

            let dgsetot_of_prob = pdg - params.g_mc() - farray[[site.0, site.1]].ln();
            let _rrate = params.k_f() * params.alpha().exp() * dgsetot_of_prob.exp();

            statesize += ntiles;
            g += dg_fill;
        } else {
            let (dg_fill, _frate, ntiles) = fill_favorable(
                farray, &mut dgatt, &mut probs, &mut state, site, params, false,
            );
            statesize += ntiles;
            g += dg_fill;
        }
    }

    TrajectoryResult {
        gcn: current_gcn,
        cn: current_cn,
        g_trace: trace,
        crit_step: current_critstep,
        size_trace,
        nucrate: current_frate,
        assemblies,
    }
}

fn probabilistic_step(
    farray: ArrayView2<f64>,
    dgatt: &mut Array2<f64>,
    probs: &mut Array2<f64>,
    state: &mut Array2<bool>,
    params: &KTAMParams,
) -> Option<((usize, usize), f64)> {
    let total_prob = probs.sum();
    if total_prob == 0.0 {
        return None;
    }

    let loc = choose_position(probs.view());
    let dg = dgatt[[loc.0, loc.1]];
    state[[loc.0, loc.1]] = true;

    update_dgatt_and_probs_around(farray, dgatt, probs, state.view(), loc, params);

    Some((loc, dg))
}

fn update_dgatt_and_probs_around(
    concmult: ArrayView2<f64>,
    dgatt: &mut Array2<f64>,
    probs: &mut Array2<f64>,
    state: ArrayView2<bool>,
    loc: (usize, usize),
    params: &KTAMParams,
) {
    let (rows, cols) = state.dim();

    for &(dx, dy) in OFF5.iter() {
        let new_x = loc.0 as i32 + dx;
        let new_y = loc.1 as i32 + dy;

        if new_x < 0 || new_x >= rows as i32 || new_y < 0 || new_y >= cols as i32 {
            continue;
        }

        let ij = (new_x as usize, new_y as usize);

        if state[[ij.0, ij.1]] {
            dgatt[[ij.0, ij.1]] = f64::INFINITY;
            probs[[ij.0, ij.1]] = 0.0;
            continue;
        }

        // Calculate bonds that would be made
        let mut b = 0;
        if ij.0 > 0 && state[[ij.0 - 1, ij.1]] {
            b += 1;
        }
        if ij.1 > 0 && state[[ij.0, ij.1 - 1]] {
            b += 1;
        }
        if ij.0 + 1 < rows && state[[ij.0 + 1, ij.1]] {
            b += 1;
        }
        if ij.1 + 1 < cols && state[[ij.0, ij.1 + 1]] {
            b += 1;
        }

        if b == 0 {
            dgatt[[ij.0, ij.1]] = f64::INFINITY;
            probs[[ij.0, ij.1]] = 0.0;
        } else {
            dgatt[[ij.0, ij.1]] =
                params.g_mc() - (b as f64) * params.g_se() - concmult[[ij.0, ij.1]].ln();
            probs[[ij.0, ij.1]] = (-dgatt[[ij.0, ij.1]]).exp();
        }
    }
}

/// Fill in favorable sites and return total dG change, forward rate, and number of tiles added
#[derive(Clone)]
pub struct GrowthResult {
    pub gce: f64,
    pub gcns: Vec<f64>,
    pub nucrate: f64,
    pub nucrates: Vec<f64>,
    pub cns: Vec<Array2<bool>>,
    pub traces: Vec<Vec<f64>>,
    pub size_traces: Vec<Vec<i64>>,
    pub stopped_pct: f64,
    pub ncns: usize,
    pub min_gcn_per_site: Array2<f64>,
    pub weight_gcn_per_site: Array2<f64>,
    pub min_size_per_site: Array2<f64>,
    pub weight_size_per_site: Array2<f64>,
    pub nucrate_per_site: Array2<f64>,
    pub min_size: usize,
    pub weight_size: f64,
    pub final_g: f64,
    pub base_g: f64,
    pub num_per_size: Vec<f64>,
    pub g_weighted_per_size: Vec<f64>,
    pub assembly_traces: Vec<Vec<Array2<bool>>>,
}

pub fn probabilistic_gce(
    farray: ArrayView2<f64>,
    trials: usize,
    params: &KTAMParams,
    depth: usize,
    calc_weightsize_g: bool,
    store_assemblies: bool,
) -> GrowthResult {
    let maxsize = farray.iter().filter(|&&x| x > 0.0).count();

    let mut found_cns = Vec::new();
    let mut found_gcns = Vec::new();
    let mut found_traces = Vec::new();
    let mut size_traces = Vec::new();
    let mut nucrates = Vec::new();

    let shape = farray.dim();
    let mut min_gcn_per_site = Array2::from_elem(shape, f64::INFINITY);
    let mut weight_gcn_per_site = Array2::from_elem(shape, f64::INFINITY);
    let mut nucrate_per_site = Array2::from_elem(shape, f64::NAN);
    let mut min_size_per_site = Array2::from_elem(shape, f64::INFINITY);
    let mut weight_size_per_site = Array2::from_elem(shape, f64::INFINITY);

    let mut sized_assemblies = vec![Vec::new(); maxsize];
    let mut sized_assembly_gs = vec![Vec::new(); maxsize];
    let mut assembly_traces = Vec::new();

    let stopped_trials = 0;
    let mut good_trials = 0;
    let mut finished_trials = 0;

    // Find the final G
    let final_g = calculate_g(farray, farray.map(|&x| x > 0.0).view(), params);
    let base_g = params.g_mc() - params.alpha();

    while finished_trials < trials {
        // Choose starting site probabilistically based on concentration
        let starting_site = choose_position(farray);

        // Get trajectory from that site
        let out = probable_trajectory(farray, starting_site, params, depth, store_assemblies);

        finished_trials += 1;

        // Store sized assemblies if requested
        if calc_weightsize_g {
            for (k, assembly) in out.assemblies.iter().enumerate() {
                if !sized_assemblies[k].contains(assembly) {
                    sized_assemblies[k].push(assembly.clone());
                    sized_assembly_gs[k].push(out.g_trace[k]);
                }
            }
        }

        found_gcns.push(out.gcn);
        found_cns.push(out.cn);
        found_traces.push(out.g_trace);
        nucrates.push(out.nucrate);
        size_traces.push(out.size_trace);
        if store_assemblies {
            assembly_traces.push(out.assemblies);
        }
        good_trials += 1;
    }
    // Remove duplicate critical nuclei
    let mut unique_indices = Vec::new();
    let mut seen = Vec::new();

    for (i, cn) in found_cns.iter().enumerate() {
        if !seen.contains(cn) {
            seen.push(cn.clone());
            unique_indices.push(i);
        }
    }

    found_gcns = unique_indices.iter().map(|&i| found_gcns[i]).collect();
    found_cns = unique_indices
        .iter()
        .map(|&i| found_cns[i].clone())
        .collect();
    found_traces = unique_indices
        .iter()
        .map(|&i| found_traces[i].clone())
        .collect();
    size_traces = unique_indices
        .iter()
        .map(|&i| size_traces[i].clone())
        .collect();
    nucrates = unique_indices.iter().map(|&i| nucrates[i]).collect();

    if store_assemblies {
        assembly_traces = unique_indices
            .iter()
            .map(|&i| assembly_traces[i].clone())
            .collect();
    }

    // Calculate Gce
    let gce = if found_gcns.is_empty() {
        f64::INFINITY
    } else {
        -found_gcns.iter().map(|&g| (-g).exp()).sum::<f64>().ln()
    };

    // Calculate per-site statistics
    for (i, cn) in found_cns.iter().enumerate() {
        for (idx, &is_present) in cn.indexed_iter() {
            if !is_present {
                continue;
            }

            let gcn = found_gcns[i];
            let exp_neg_gcn = (-gcn).exp();

            min_gcn_per_site[idx] = min_gcn_per_site[idx].min(gcn);
            weight_gcn_per_site[idx] = if weight_gcn_per_site[idx].is_infinite() {
                gcn
            } else {
                (weight_gcn_per_site[idx] + gcn * exp_neg_gcn) / (1.0 + exp_neg_gcn)
            };

            let size = calculate_n(cn.view());
            min_size_per_site[idx] = min_size_per_site[idx].min(size as f64);
            weight_size_per_site[idx] = if weight_size_per_site[idx].is_infinite() {
                size as f64
            } else {
                (weight_size_per_site[idx] + (size as f64) * exp_neg_gcn) / (1.0 + exp_neg_gcn)
            };

            nucrate_per_site[idx] = if nucrate_per_site[idx].is_nan() {
                nucrates[i] / (size as f64)
            } else {
                nucrate_per_site[idx] + nucrates[i] / (size as f64)
            };
        }
    }

    // Calculate size statistics
    let min_size = found_cns
        .iter()
        .map(|cn| calculate_n(cn.view()))
        .min()
        .unwrap_or(0);

    let weight_size = if found_gcns.is_empty() {
        0.0
    } else {
        let sizes: Vec<f64> = found_cns
            .iter()
            .map(|cn| calculate_n(cn.view()) as f64)
            .collect();
        let exp_neg_gcns: Vec<f64> = found_gcns.iter().map(|&g| (-g).exp()).collect();
        let total_exp = exp_neg_gcns.iter().sum::<f64>();

        sizes
            .iter()
            .zip(exp_neg_gcns.iter())
            .map(|(&s, &e)| s * e)
            .sum::<f64>()
            / total_exp
    };

    // Calculate per-size statistics
    let mut num_per_size = vec![0.0; maxsize];
    let mut g_weighted_per_size = vec![0.0; maxsize];

    for (i, size) in size_traces.iter().enumerate() {
        for (&s, &g) in size.iter().zip(found_traces[i].iter()) {
            let idx = (s as usize) - 1;
            if idx < maxsize {
                num_per_size[idx] += 1.0;
                g_weighted_per_size[idx] += g;
            }
        }
    }

    for i in 0..maxsize {
        if num_per_size[i] > 0.0 {
            g_weighted_per_size[i] /= num_per_size[i];
        }
    }

    GrowthResult {
        gce,
        ncns: found_cns.len(),
        gcns: found_gcns,
        nucrate: nucrates.iter().sum(),
        nucrates,
        cns: found_cns,
        traces: found_traces,
        size_traces,
        stopped_pct: stopped_trials as f64 / trials as f64,
        min_gcn_per_site,
        weight_gcn_per_site,
        min_size_per_site,
        weight_size_per_site,
        nucrate_per_site,
        min_size,
        weight_size,
        final_g,
        base_g,
        num_per_size,
        g_weighted_per_size,
        assembly_traces,
    }
}

fn fill_favorable(
    concmult: ArrayView2<f64>,
    dgatt: &mut Array2<f64>,
    probs: &mut Array2<f64>,
    state: &mut Array2<bool>,
    loc: (usize, usize),
    params: &KTAMParams,
    is_current_critnuc: bool,
) -> (f64, f64, usize) {
    let mut total_dg = 0.0;
    let mut ntiles = 0;

    // Calculate forward rate if this is current critical nucleus
    let mut frate = 0.0;
    if is_current_critnuc {
        for &(dx, dy) in OFF4.iter() {
            let new_x = loc.0 as i32 + dx;
            let new_y = loc.1 as i32 + dy;

            let (rows, cols) = state.dim();
            if new_x < 0 || new_x >= rows as i32 || new_y < 0 || new_y >= cols as i32 {
                continue;
            }

            let trial_site = (new_x as usize, new_y as usize);
            if dgatt[[trial_site.0, trial_site.1]] < 0.0 {
                frate += params.k_f() * (-params.g_mc() + params.alpha()).exp();
            }
        }
    }

    // Keep filling until no more favorable sites
    loop {
        let mut did_something = false;

        // Check sites adjacent to previous attachments
        for &(dx, dy) in OFF4.iter() {
            let new_x = loc.0 as i32 + dx;
            let new_y = loc.1 as i32 + dy;

            let (rows, cols) = state.dim();
            if new_x < 0 || new_x >= rows as i32 || new_y < 0 || new_y >= cols as i32 {
                continue;
            }

            let trial_site = (new_x as usize, new_y as usize);
            let dg = dgatt[[trial_site.0, trial_site.1]];

            // Found a favorable site (dG < 0)
            if dg < 0.0 {
                state[[trial_site.0, trial_site.1]] = true;
                ntiles += 1;
                total_dg += dg;

                update_dgatt_and_probs_around(
                    concmult,
                    dgatt,
                    probs,
                    state.view(),
                    trial_site,
                    params,
                );

                did_something = true;
                break;
            }
        }

        if !did_something {
            break;
        }
    }

    (total_dg, frate, ntiles)
}

use ndarray::s;

/// The "Window Score"
/// This is essentially the log of the sum of the product of concentrations
/// in k x k squares, or "windows".
pub fn window_score(conc_array: ArrayView2<f64>, k: usize) -> f64 {
    let log_z = conc_array.mapv(|x| (x as f64).ln());

    let mut convolved = Array2::from_elem((conc_array.dim().0 - k + 1, conc_array.dim().1 - k + 1), 0.0);

    convolved
        .indexed_iter_mut()
        .for_each(|((i, j), x)| *x = log_z.slice(s![i..i + k, j..j + k]).fold(0.0, |a, &b| a + b));

    let v: Vec<_> = convolved.into_iter().collect();
    v.into_iter().ln_sum_exp()
}

/// The "Window Score"
/// This is essentially the log of the sum of the product of concentrations
/// in k x k squares, or "windows".
pub fn window_k_array_with_bonds(
    conc_array: ArrayView2<f64>,
    gse_to_e: ArrayView2<f64>,
    gse_to_s: ArrayView2<f64>,
    k: usize,
) -> Array2<f64> {
    // Check dimensions of gse arrays match conc_array
    let (rows, cols) = conc_array.dim();
    assert!(
        (gse_to_e.dim() == (rows, cols) || gse_to_e.dim() == (rows - 1, cols - 1))
            && (gse_to_s.dim() == (rows, cols) || gse_to_s.dim() == (rows - 1, cols - 1)),
        "gse arrays must be same size as conc_array or 1 smaller in each dimension"
    );

    let mut convolved = Array2::<f64>::from_elem(
        (conc_array.dim().0 - k + 1, conc_array.dim().1 - k + 1),
        INFINITY,
    );

    convolved.indexed_iter_mut().for_each(|((i, j), x)| {
        let mut first = true;
        let mut b = 0;
        conc_array
            .slice(s![i..i + k, j..j + k])
            .indexed_iter()
            .for_each(|((ii, jj), z)| {
                if *z > 0.0 {
                    if first {
                        first = false;
                        *x = 0.0;
                    }
                    let mut v = -(z.ln());
                    if (ii < k - 1) && (conc_array[[i + ii + 1, j + jj]] > 0.0) {
                        v -= gse_to_s[[i + ii, j + jj]];
                    };
                    if (jj < k - 1) && (conc_array[[i + ii, j + jj + 1]] > 0.0) {
                        v -= gse_to_e[[i + ii, j + jj]];
                    };
                    *x += v;
                }
            });
    });

    convolved
}

// pub fn window_score_pat(pat: &[(usize, usize)], m: f64, base: f64, k: usize) -> f64 {
//     let mut z = Array2::from_elem((16, 16), base);
//     for &(x, y) in pat {
//         z[[x + 5, y + 5]] = m * base;
//     }
//     window_score(z.view(), k)
// }

pub fn window_nuc_k(z: ArrayView2<f64>, k: usize, gse: f64, alpha: f64, base: f64, kf: f64) -> f64 {
    let b = 2 * k * (k - 1);
    let p = 4 * k;

    let score = window_score(z, k);
    kf * (p as f64) * base * (score + ((b as f64) * gse) - (((k * k) - 1) as f64 * alpha)).exp()
}

pub fn window_nuc_rate(
    z: ArrayView2<f64>,
    gse: f64,
    alpha: f64,
    kf: f64,
    krange: std::ops::Range<usize>,
) -> f64 {
    // use median of nonzero z values as base
    let mut b = z.iter().filter(|&&x| x > 0.0).cloned().collect::<Vec<_>>();
    b.sort_unstable_by(move |&i, &j| i.partial_cmp(&j).unwrap());
    let base = b[b.len() / 2];

    krange
        .map(|k| window_nuc_k(z, k, gse, alpha, base, kf))
        .fold(f64::INFINITY, |a, b| a.min(b))
}

pub fn window_path_nuc_rate(concarray: ArrayView2<f64>, gse: f64, alpha: f64, kf: f64, include_size_in_forward: bool) -> f64 {
    // use median of nonzero z values as base
    let mut b = concarray.iter().filter(|&&x| x > 0.0).cloned().collect::<Vec<_>>();
    b.sort_unstable_by(move |&i, &j| i.partial_cmp(&j).unwrap());
    let base = b[b.len() / 2];

    let z = concarray.map(|x| x * (-alpha).exp());



    let bondarray = Array2::from_elem(z.dim(), gse);
    let window_arrays = (1..=(z.dim().0))
        .map(|x| window_k_array_with_bonds(z.view(), bondarray.view(), bondarray.view(), x));
    // Generate with pruning
    let window_arrays: Vec<_> = window_arrays.collect();
    let maxs: Vec<_> = window_arrays
        .iter()
        .map(|v| v.fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
        .collect();
    let mins: Vec<_> = window_arrays
        .iter()
        .map(|v| v.fold(f64::INFINITY, |a, &b| a.min(b)))
        .collect();

    let mut startk = 0;
    let mut stopk = window_arrays.len() - 1;

    while maxs[startk..stopk]
        .iter()
        .zip(mins[startk + 1..].iter())
        .any(|(&m1, &m2)| m1 < m2)
    {
        startk += 1;
    }
    while mins[startk..stopk - 1]
        .iter()
        .zip(maxs[startk + 1..].iter())
        .any(|(&m1, &m2)| m1 > m2)
    {
        stopk -= 1;
    }

    let mut paths: Vec<(Vec<(usize, usize)>, f64)> = iproduct!(
        0..window_arrays[stopk].dim().0,
        0..window_arrays[stopk].dim().1
    )
    .map(|(i, j)| (vec![(i, j)], window_arrays[stopk][[i, j]]))
    .collect();

    for k in (startk..stopk).rev() {
        let mut newpaths = Vec::new();
        let mut newends = HashMap::new();

        for path in paths {
            let (i, j) = path.0[0];
            for (di, dj) in [(0, 1), (1, 0), (1, 1), (0, 0)].iter() {
                let m = window_arrays[k][[i + di, j + dj]];
                if m > path.1 {
                    newends.insert((i + di, j + dj), m);
                } else {
                    let mut new_path = vec![(i + di, j + dj)];
                    new_path.extend(path.0.iter());
                    newpaths.push((new_path, path.1));
                }
            }
        }

        paths = newends
            .into_iter()
            .map(|((i, j), m)| (vec![(i, j)], m))
            .chain(newpaths)
            .collect();
    }


    // Convert paths into arrays of values along each path
    let pathvals: Vec<Vec<f64>> = paths
        .iter()
        .map(|(path, _)| {
            path.iter()
                .enumerate()
                .map(|(k, &(i, j))| window_arrays[k + startk][[i, j]])
                .collect()
        })
        .collect();

    // Find index of maximum value in each path
    let pathcritsize: Vec<usize> = pathvals
        .iter()
        .map(|vals| {
            vals.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        })
        .collect();

    // Create critical points from indices and paths
    let pathcrits: Vec<(usize, (usize, usize))> = pathcritsize
        .iter()
        .zip(paths.iter())
        .map(|(&k, (path, _))| (k + startk, path[k]))
        .collect();

    // Create array of zeros for each window size
    let mut iscrit: Vec<Array2<f64>> = window_arrays
        .iter()
        .map(|arr| Array2::zeros(arr.dim()))
        .collect();

    // Mark critical points
    for (k, (i, j)) in pathcrits {
        iscrit[k][[i, j]] = 1.0;
    }

    let mut nucrate = 0.0;
    // sum(np.sum(np.exp(-v+alpha) * ic * kf * np.exp(-gmc+alpha)) for v, ic in zip(vv, iscrit))

    let mut firstprint = true;

    window_arrays.iter().zip(iscrit.iter()).enumerate().for_each(|(k, (v, ic))| {
        ndarray::Zip::from(v).and(ic).for_each(|v, ic| {
            let c = ((-v + alpha) as  f128).exp() * (*ic as f128) * kf as f128 * base as f128 * (if include_size_in_forward { 4.0*(k+1) as f128 } else { 1.0 });
            let c = c as f64;
            nucrate += c;
        });
    });

    nucrate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_traits::ToPrimitive;

    #[test]
    fn test_window_score() {
        // Test 1: k x k array of ones
        let k = 3;
        let ones = Array2::ones((k, k));
        assert_relative_eq!(window_score(ones.view(), k), 0.0, epsilon = 1e-10);

        // Test 2: k x k square of ones in larger array
        let mut larger = Array2::zeros((5, 5));
        larger.slice_mut(s![1..4, 1..4]).fill(1.0);
        assert_relative_eq!(window_score(larger.view(), k), 0.0, epsilon = 1e-10);

        // Test 3: k x k array of value X
        let x: f64 = 2.0;
        let x_array = Array2::from_elem((k, k), x);
        let expected = (k * k) as f64 * x.ln();
        assert_relative_eq!(window_score(x_array.view(), k), expected, epsilon = 1e-10);
    }

    // Given an N x N array of V, the window score with size k will be ln((N-k+1)^2 * v^k)
    #[test]
    fn test_window_score_2() {
        let n: usize = 20;
        let k = 5;
        let v = 1e-7;
        let expected = (k as f64).powi(2) * v.ln() + 2.0 * ((n - k + 1) as f64).ln();
        let mut array = Array2::from_elem((n, n), v);
        assert_relative_eq!(window_score(array.view(), k), expected, epsilon = 1e-10);
    }
}
