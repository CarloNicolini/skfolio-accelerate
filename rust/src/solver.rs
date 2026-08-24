//! Per-worker Clarabel solver with update_data / update_P.

#![allow(non_snake_case)]

use std::sync::Arc;

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus};

use crate::problem::ProblemPattern;
use crate::result::SolveRecord;

pub fn default_settings(solver_threads: u32) -> DefaultSettings<f64> {
    let mut settings = DefaultSettings::<f64>::default();
    settings.verbose = false;
    settings.presolve_enable = false;
    settings.input_sparse_dropzeros = false;
    settings.max_threads = solver_threads.max(1);
    settings.tol_gap_abs = 1e-9;
    settings.tol_gap_rel = 1e-9;
    settings
}

pub struct WorkerSolver {
    pattern: Arc<ProblemPattern>,
    settings: DefaultSettings<f64>,
    solver: Option<DefaultSolver<f64>>,
    p_buf: Vec<f64>,
    q_buf: Vec<f64>,
    a_buf: Vec<f64>,
    b_buf: Vec<f64>,
}

impl WorkerSolver {
    pub fn new(pattern: Arc<ProblemPattern>, solver_threads: u32) -> Self {
        Self {
            pattern,
            settings: default_settings(solver_threads),
            solver: None,
            p_buf: Vec::new(),
            q_buf: Vec::new(),
            a_buf: Vec::new(),
            b_buf: Vec::new(),
        }
    }

    fn fill(buf: &mut Vec<f64>, src: &[f64]) {
        buf.clear();
        buf.extend_from_slice(src);
    }

    pub fn solve_instance(
        &mut self,
        p_nz: &[f64],
        q: &[f64],
        a_nz: &[f64],
        b: &[f64],
        update_q: bool,
        update_a: bool,
    ) -> Result<SolveRecord, String> {
        if self.solver.is_none() {
            let P = CscMatrix::new(
                self.pattern.n_vars,
                self.pattern.n_vars,
                self.pattern.p_indptr.clone(),
                self.pattern.p_indices.clone(),
                p_nz.to_vec(),
            );
            let A = CscMatrix::new(
                self.pattern.n_cons,
                self.pattern.n_vars,
                self.pattern.a_indptr.clone(),
                self.pattern.a_indices.clone(),
                a_nz.to_vec(),
            );
            let solver = DefaultSolver::new(
                &P,
                &q.to_vec(),
                &A,
                &b.to_vec(),
                &self.pattern.cones,
                self.settings.clone(),
            )
            .map_err(|err| format!("Clarabel setup failed: {err:?}"))?;
            self.solver = Some(solver);
        } else {
            Self::fill(&mut self.p_buf, p_nz);
            let solver = self.solver.as_mut().unwrap();
            solver
                .update_P(&self.p_buf)
                .map_err(|err| format!("Clarabel update_P failed: {err}"))?;
            if update_q {
                Self::fill(&mut self.q_buf, q);
                solver
                    .update_q(&self.q_buf)
                    .map_err(|err| format!("Clarabel update_q failed: {err}"))?;
            }
            if update_a {
                Self::fill(&mut self.a_buf, a_nz);
                Self::fill(&mut self.b_buf, b);
                solver
                    .update_A(&self.a_buf)
                    .map_err(|err| format!("Clarabel update_A failed: {err}"))?;
                solver
                    .update_b(&self.b_buf)
                    .map_err(|err| format!("Clarabel update_b failed: {err}"))?;
            }
        }

        let solver = self.solver.as_mut().unwrap();
        solver.solve();
        Ok(record_from_solver(solver, &self.pattern))
    }
}

fn record_from_solver(solver: &DefaultSolver<f64>, pattern: &ProblemPattern) -> SolveRecord {
    let status = format!("{:?}", solver.solution.status);
    let solved = matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    );
    let start = pattern.weight_start.min(solver.solution.x.len());
    let end = (pattern.weight_start + pattern.weight_len).min(solver.solution.x.len());
    SolveRecord {
        x: solver.solution.x[start..end].to_vec(),
        status,
        obj: if solved {
            solver.solution.obj_val
        } else {
            f64::NAN
        },
        iterations: solver.solution.iterations as i64,
        solve_time: solver.solution.solve_time,
    }
}
