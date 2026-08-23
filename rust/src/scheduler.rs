//! Rayon worker pool: one Clarabel solver per thread.

use rayon::prelude::*;

use crate::problem::ProblemPattern;
use crate::result::SolveRecord;
use crate::solver::WorkerSolver;

pub fn solve_batch(
    pattern: &ProblemPattern,
    p_nz: &[Vec<f64>],
    q: &[Vec<f64>],
    a_nz: &[Vec<f64>],
    b: &[Vec<f64>],
    n_jobs: usize,
    solver_threads: u32,
) -> Result<Vec<SolveRecord>, String> {
    let n = p_nz.len();
    if q.len() != n || a_nz.len() != n || b.len() != n {
        return Err("solve_many input rows must all have the same batch size".into());
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let workers = n_jobs.max(1).min(n);
    let chunk_size = (n + workers - 1) / workers;
    let indexed: Vec<usize> = (0..n).collect();

    let parts: Result<Vec<Vec<(usize, SolveRecord)>>, String> = indexed
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut solver = WorkerSolver::new(pattern.clone(), solver_threads);
            let mut out = Vec::with_capacity(chunk.len());
            for &i in chunk {
                let rec = solver.solve_instance(&p_nz[i], &q[i], &a_nz[i], &b[i])?;
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
