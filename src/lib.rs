use ndarray::{Array2, ArrayView2, Axis};
use rand::Rng;
use std::f64;

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArray2};
#[cfg(feature = "python")]
use pyo3::{prelude::*, types::PyDict};

#[cfg(feature = "python")]
#[pymodule]
fn two_step_nuc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_probabilistic_gce, m)?)?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn py_probabilistic_gce(
    py: Python<'_>,
    farray: &PyArray2<f64>,
    trials: usize,
    params: KTAMParams,
    depth: usize,
    calc_weightsize_g: bool,
    store_assemblies: bool,
) -> PyResult<PyObject> {
    let farray = unsafe { farray.as_array() }.to_owned();
    
    let result = probabilistic_gce(
        &farray,
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
    dict.set_item("weight_gcn_per_site", result.weight_gcn_per_site.into_pyarray(py))?;
    dict.set_item("min_size_per_site", result.min_size_per_site.into_pyarray(py))?;
    dict.set_item("weight_size_per_site", result.weight_size_per_site.into_pyarray(py))?;
    dict.set_item("nucrate_per_site", result.nucrate_per_site.into_pyarray(py))?;
    dict.set_item("min_size", result.min_size)?;
    dict.set_item("weight_size", result.weight_size)?;
    dict.set_item("final_g", result.final_g)?;
    dict.set_item("base_g", result.base_g)?;
    dict.set_item("num_per_size", result.num_per_size)?;
    dict.set_item("g_weighted_per_size", result.g_weighted_per_size)?;

    Ok(dict.into())
}

pub const OFF4: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub const OFF5: [(i32, i32); 5] = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)];

#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub struct KTAMParams {
    #[pyo3(get, set)]
    gmc: f64,
    #[pyo3(get, set)]
    gse: f64,
    #[pyo3(get, set)]
    alpha: f64,
    #[pyo3(get, set)]
    kf: f64,
}

impl KTAMParams {
    pub fn new(gmc: f64, gse: f64, alpha: f64, kf: f64) -> Self {
        Self { gmc, gse, alpha, kf }
    }

    pub fn g_se(&self) -> f64 { self.gse }
    pub fn g_mc(&self) -> f64 { self.gmc }
    pub fn k_f(&self) -> f64 { self.kf }
    pub fn epsilon(&self) -> f64 { 2.0 * self.gse - self.gmc }
    pub fn alpha(&self) -> f64 { self.alpha }
    pub fn kh_f(&self) -> f64 { self.kf * (-self.alpha).exp() }
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
fn choose_position(weights: &Array2<f64>) -> (usize, usize) {
    let mut rng = rand::thread_rng();
    let total = weights.sum();
    let mut trigger = total * rng.gen::<f64>();
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
fn calculate_n(state: &Array2<bool>) -> usize {
    state.iter().filter(|&&x| x).count()
}

fn calculate_b(state: &Array2<bool>) -> usize {
    let (rows, cols) = state.dim();
    let mut bonds = 0;
    
    // Horizontal bonds
    for i in 0..rows {
        for j in 0..(cols-1) {
            if state[[i, j]] && state[[i, j+1]] {
                bonds += 1;
            }
        }
    }
    
    // Vertical bonds
    for i in 0..(rows-1) {
        for j in 0..cols {
            if state[[i, j]] && state[[i+1, j]] {
                bonds += 1;
            }
        }
    }
    
    bonds
}

fn calculate_g(conc_mult: &Array2<f64>, state: &Array2<bool>, params: &KTAMParams) -> f64 {
    let n = calculate_n(state) as f64;
    let b = calculate_b(state) as f64;
    let log_sum: f64 = state.iter()
        .zip(conc_mult.iter())
        .filter(|(&s, _)| s)
        .map(|(_, &c)| c.ln())
        .sum();
    
    n * params.g_mc() - b * params.g_se() - log_sum - params.alpha()
}

/// Calculate a single SGM trajectory
pub fn probable_trajectory(
    farray: &Array2<f64>,
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
    
    update_dgatt_and_probs_around(farray, &mut dgatt, &mut probs, &state, starting_site, params);

    while stepnum <= max_steps {
        let (site, dg) = match probabilistic_step(farray, &mut dgatt, &mut probs, &mut state, params) {
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
                farray, 
                &mut dgatt, 
                &mut probs, 
                &mut state, 
                site,
                params,
                true
            );
            current_frate = frate * (-current_gcn).exp();
            
            let dgsetot_of_prob = pdg - params.g_mc() - farray[[site.0, site.1]].ln();
            let _rrate = params.k_f() * params.alpha().exp() * dgsetot_of_prob.exp();
            
            statesize += ntiles;
            g += dg_fill;
        } else {
            let (dg_fill, _frate, ntiles) = fill_favorable(
                farray,
                &mut dgatt,
                &mut probs,
                &mut state,
                site,
                params,
                false
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
    farray: &Array2<f64>,
    dgatt: &mut Array2<f64>,
    probs: &mut Array2<f64>,
    state: &mut Array2<bool>,
    params: &KTAMParams,
) -> Option<((usize, usize), f64)> {
    let total_prob = probs.sum();
    if total_prob == 0.0 {
        return None;
    }

    let loc = choose_position(probs);
    let dg = dgatt[[loc.0, loc.1]];
    state[[loc.0, loc.1]] = true;

    update_dgatt_and_probs_around(farray, dgatt, probs, state, loc, params);

    Some((loc, dg))
}

fn update_dgatt_and_probs_around(
    concmult: &Array2<f64>,
    dgatt: &mut Array2<f64>,
    probs: &mut Array2<f64>,
    state: &Array2<bool>,
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
        if ij.0 > 0 && state[[ij.0 - 1, ij.1]] { b += 1; }
        if ij.1 > 0 && state[[ij.0, ij.1 - 1]] { b += 1; }
        if ij.0 + 1 < rows && state[[ij.0 + 1, ij.1]] { b += 1; }
        if ij.1 + 1 < cols && state[[ij.0, ij.1 + 1]] { b += 1; }

        if b == 0 {
            dgatt[[ij.0, ij.1]] = f64::INFINITY;
            probs[[ij.0, ij.1]] = 0.0;
        } else {
            dgatt[[ij.0, ij.1]] = params.g_mc() - (b as f64) * params.g_se() - concmult[[ij.0, ij.1]].ln();
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
    farray: &Array2<f64>,
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

    let mut stopped_trials = 0;
    let mut good_trials = 0;
    let mut finished_trials = 0;

    // Find the final G
    let final_g = calculate_g(farray, &Array2::from_elem(shape, true), params);
    let base_g = params.g_mc() - params.alpha();

    while finished_trials < trials {
        // Choose starting site probabilistically based on concentration
        let starting_site = choose_position(farray);
        
        // Get trajectory from that site
        let out = probable_trajectory(
            farray,
            starting_site,
            params,
            depth,
            store_assemblies,
        );

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

    // Calculate Gce
    let gce = if found_gcns.is_empty() {
        f64::INFINITY
    } else {
        -(-found_gcns.iter().sum::<f64>()).exp().ln()
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
            
            let size = calculate_n(cn);
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
    let min_size = found_cns.iter()
        .map(|cn| calculate_n(cn))
        .min()
        .unwrap_or(0);
        
    let weight_size = if found_gcns.is_empty() {
        0.0
    } else {
        let sizes: Vec<f64> = found_cns.iter()
            .map(|cn| calculate_n(cn) as f64)
            .collect();
        let exp_neg_gcns: Vec<f64> = found_gcns.iter()
            .map(|&g| (-g).exp())
            .collect();
        let total_exp = exp_neg_gcns.iter().sum::<f64>();
        
        sizes.iter()
            .zip(exp_neg_gcns.iter())
            .map(|(&s, &e)| s * e)
            .sum::<f64>() / total_exp
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
    concmult: &Array2<f64>,
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
                    state,
                    trial_site,
                    params
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
