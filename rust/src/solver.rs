//! Per-worker Clarabel solver with update_data.

#![allow(non_snake_case)]

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
    pattern: ProblemPattern,
    settings: DefaultSettings<f64>,
    solver: Option<DefaultSolver<f64>>,
}

impl WorkerSolver {
    pub fn new(pattern: ProblemPattern, solver_threads: u32) -> Self {
        Self {
            pattern,
            settings: default_settings(solver_threads),
            solver: None,
        }
    }

    pub fn solve_instance(
        &mut self,
        p_nz: &[f64],
        q: &[f64],
        a_nz: &[f64],
        b: &[f64],
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
            let solver = self.solver.as_mut().unwrap();
            solver
                .update_data(&p_nz.to_vec(), &q.to_vec(), &a_nz.to_vec(), &b.to_vec())
                .map_err(|err| format!("Clarabel update_data failed: {err}"))?;
        }

        let solver = self.solver.as_mut().unwrap();
        solver.solve();
        Ok(record_from_solver(solver))
    }
}

fn record_from_solver(solver: &DefaultSolver<f64>) -> SolveRecord {
    let status = format!("{:?}", solver.solution.status);
    let solved = matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    );
    SolveRecord {
        x: solver.solution.x.clone(),
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
