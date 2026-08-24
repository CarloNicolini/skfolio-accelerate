//! Problem topology: CSC patterns and Clarabel cones.

#![allow(non_snake_case)]

use clarabel::solver::{NonnegativeConeT, SecondOrderConeT, SupportedConeT, ZeroConeT};

#[derive(Clone)]
pub struct ProblemPattern {
    pub n_vars: usize,
    pub n_cons: usize,
    pub p_indptr: Vec<usize>,
    pub p_indices: Vec<usize>,
    pub a_indptr: Vec<usize>,
    pub a_indices: Vec<usize>,
    pub cones: Vec<SupportedConeT<f64>>,
    pub weight_start: usize,
    pub weight_len: usize,
}

pub fn cones_from_spec(kinds: &[String], dims: &[i64]) -> Result<Vec<SupportedConeT<f64>>, String> {
    if kinds.len() != dims.len() {
        return Err("cone_kinds and cone_dims must have the same length".into());
    }
    let mut cones = Vec::with_capacity(kinds.len());
    for (kind, dim) in kinds.iter().zip(dims.iter()) {
        let n = *dim as usize;
        match kind.to_ascii_lowercase().as_str() {
            "zero" => cones.push(ZeroConeT(n)),
            "nonnegative" | "nonneg" => cones.push(NonnegativeConeT(n)),
            "soc" | "secondorder" => cones.push(SecondOrderConeT(n)),
            other => return Err(format!("unsupported cone kind {other}")),
        }
    }
    Ok(cones)
}
