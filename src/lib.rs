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
