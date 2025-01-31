use ndarray::{Array2, ArrayView2, Axis};
use rand::Rng;
use std::f64;

pub const OFF4: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub const OFF5: [(i32, i32); 5] = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)];

#[derive(Clone, Debug)]
pub struct KTAMParams {
    gmc: f64,
    gse: f64,
    alpha: f64,
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
