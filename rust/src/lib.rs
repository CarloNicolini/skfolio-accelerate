//! PyO3 bindings for the Clarabel update engine.

#![allow(non_snake_case)]

use std::sync::Mutex;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

mod problem;
mod result;
mod scheduler;
mod solver;

use problem::{cones_from_spec, ProblemPattern};
use scheduler::solve_batch;
use solver::WorkerSolver;

fn i64_to_usize(values: &[i64], name: &str) -> PyResult<Vec<usize>> {
    values
        .iter()
        .map(|&v| {
            usize::try_from(v).map_err(|_| {
                PyValueError::new_err(format!("{name} contains a negative index"))
            })
        })
        .collect()
}

fn rows(matrix: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f64>>> {
    let view = matrix.as_array();
    Ok(view.outer_iter().map(|row| row.to_vec()).collect())
}

#[pyclass]
struct ExecutionEngine {
    pattern: ProblemPattern,
    n_jobs: usize,
    solver_threads: u32,
    sequential: Mutex<Option<WorkerSolver>>,
}

#[pymethods]
impl ExecutionEngine {
    #[new]
    #[pyo3(signature = (
        n_vars,
        n_cons,
        p_indptr,
        p_indices,
        a_indptr,
        a_indices,
        cone_kinds,
        cone_dims,
        weight_start,
        weight_len,
        n_jobs = 1,
        solver_threads = 1,
    ))]
    fn new(
        n_vars: usize,
        n_cons: usize,
        p_indptr: PyReadonlyArray1<i64>,
        p_indices: PyReadonlyArray1<i64>,
        a_indptr: PyReadonlyArray1<i64>,
        a_indices: PyReadonlyArray1<i64>,
        cone_kinds: Vec<String>,
        cone_dims: PyReadonlyArray1<i64>,
        weight_start: usize,
        weight_len: usize,
        n_jobs: usize,
        solver_threads: u32,
    ) -> PyResult<Self> {
        let cones = cones_from_spec(&cone_kinds, &cone_dims.as_slice()?.to_vec())
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            pattern: ProblemPattern {
                n_vars,
                n_cons,
                p_indptr: i64_to_usize(p_indptr.as_slice()?, "p_indptr")?,
                p_indices: i64_to_usize(p_indices.as_slice()?, "p_indices")?,
                a_indptr: i64_to_usize(a_indptr.as_slice()?, "a_indptr")?,
                a_indices: i64_to_usize(a_indices.as_slice()?, "a_indices")?,
                cones,
                weight_start,
                weight_len,
            },
            n_jobs: n_jobs.max(1),
            solver_threads: solver_threads.max(1),
            sequential: Mutex::new(None),
        })
    }

    fn solve_many<'py>(
        &self,
        py: Python<'py>,
        p_nz: PyReadonlyArray2<f64>,
        q: PyReadonlyArray2<f64>,
        a_nz: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyList>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let p_rows = rows(p_nz)?;
        let q_rows = rows(q)?;
        let a_rows = rows(a_nz)?;
        let b_rows = rows(b)?;
        let pattern = self.pattern.clone();
        let n_jobs = self.n_jobs;
        let solver_threads = self.solver_threads;

        let records = py
            .allow_threads(|| -> Result<_, String> {
                if n_jobs <= 1 || p_rows.len() <= 1 {
                    let mut guard = self
                        .sequential
                        .lock()
                        .map_err(|err| format!("solver mutex poisoned: {err}"))?;
                    if guard.is_none() {
                        *guard = Some(WorkerSolver::new(pattern.clone(), solver_threads));
                    }
                    let worker = guard.as_mut().unwrap();
                    let mut out = Vec::with_capacity(p_rows.len());
                    for i in 0..p_rows.len() {
                        out.push(worker.solve_instance(
                            &p_rows[i],
                            &q_rows[i],
                            &a_rows[i],
                            &b_rows[i],
                        )?);
                    }
                    Ok(out)
                } else {
                    solve_batch(
                        &pattern,
                        &p_rows,
                        &q_rows,
                        &a_rows,
                        &b_rows,
                        n_jobs,
                        solver_threads,
                    )
                }
            })
            .map_err(PyValueError::new_err)?;

        let n = records.len();
        let n_vars = self.pattern.n_vars;
        let mut xs = vec![0.0; n * n_vars];
        let mut objs = vec![0.0; n];
        let mut iters = vec![0_i64; n];
        let mut times = vec![0.0; n];
        let mut statuses = Vec::with_capacity(n);
        for (i, rec) in records.iter().enumerate() {
            let len = rec.x.len().min(n_vars);
            xs[i * n_vars..i * n_vars + len].copy_from_slice(&rec.x[..len]);
            objs[i] = rec.obj;
            iters[i] = rec.iterations;
            times[i] = rec.solve_time;
            statuses.push(rec.status.clone());
        }

        let xs_arr = PyArray2::from_vec2_bound(py, &xs.chunks(n_vars).map(|r| r.to_vec()).collect::<Vec<_>>())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let status_list = PyList::new_bound(py, statuses);
        let objs_arr = PyArray1::from_vec_bound(py, objs);
        let iters_arr = PyArray1::from_vec_bound(py, iters);
        let times_arr = PyArray1::from_vec_bound(py, times);
        Ok((xs_arr, status_list, objs_arr, iters_arr, times_arr))
    }
}

#[pymodule]
fn _skfolio_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExecutionEngine>()?;
    Ok(())
}
