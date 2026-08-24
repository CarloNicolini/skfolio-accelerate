//! Rayon worker pool: one persistent Clarabel solver per worker.

use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::problem::ProblemPattern;
use crate::result::SolveRecord;
use crate::solver::WorkerSolver;

fn row(data: &[f64], i: usize, stride: usize, shared: bool) -> &[f64] {
    let start = if shared { 0 } else { i * stride };
    &data[start..start + stride]
}

pub fn solve_batch(
    pattern: &Arc<ProblemPattern>,
    workers: &[Mutex<Option<WorkerSolver>>],
    p_nz: &[f64],
    p_stride: usize,
    q: &[f64],
    q_stride: usize,
    q_shared: bool,
    a_nz: &[f64],
    a_stride: usize,
    a_shared: bool,
    b: &[f64],
    b_stride: usize,
    b_shared: bool,
    n: usize,
    solver_threads: u32,
) -> Result<Vec<SolveRecord>, String> {
    if p_stride == 0 || q_stride == 0 || a_stride == 0 || b_stride == 0 {
        return Err("solve_many strides must be positive".into());
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let n_workers = workers.len().max(1).min(n);
    let chunk_size = (n + n_workers - 1) / n_workers;

    let parts: Result<Vec<Vec<(usize, SolveRecord)>>, String> = (0..n_workers)
        .into_par_iter()
        .map(|wid| {
            let start = wid * chunk_size;
            if start >= n {
                return Ok(Vec::new());
            }
            let end = (start + chunk_size).min(n);
            let mut guard = workers[wid]
                .lock()
                .map_err(|err| format!("solver mutex poisoned: {err}"))?;
            if guard.is_none() {
                *guard = Some(WorkerSolver::new(Arc::clone(pattern), solver_threads));
            }
            let solver = guard.as_mut().unwrap();
            let mut out = Vec::with_capacity(end - start);
            for i in start..end {
                let rec = solver.solve_instance(
                    row(p_nz, i, p_stride, false),
                    row(q, i, q_stride, q_shared),
                    row(a_nz, i, a_stride, a_shared),
                    row(b, i, b_stride, b_shared),
                    !q_shared || i == start,
                    !a_shared || i == start,
                )?;
                out.push((i, rec));
            }
            Ok(out)
        })
        .collect();

    let mut ordered = vec![None; n];
    for (i, rec) in parts?.into_iter().flatten() {
        ordered[i] = Some(rec);
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(i, rec)| rec.ok_or_else(|| format!("missing result {i}")))
        .collect()
}
