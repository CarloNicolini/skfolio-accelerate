//! PyO3 bindings for the Clarabel update engine.

#![allow(non_snake_case)]

use std::sync::{Arc, Mutex};

use numpy::ndarray::Array2;
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

fn contig_dims(
    view: numpy::ndarray::ArrayView2<'_, f64>,
    name: &str,
) -> PyResult<(usize, usize)> {
    if !view.is_standard_layout() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous float64"
        )));
    }
    Ok((view.nrows(), view.ncols()))
}

fn shared_or_batch(n_batch: usize, n_rows: usize, name: &str) -> PyResult<bool> {
    if n_rows == n_batch {
        Ok(false)
    } else if n_rows == 1 {
        Ok(true)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must have {n_batch} rows or a single shared row, got {n_rows}"
        )))
    }
}

#[pyclass]
struct ExecutionEngine {
    pattern: Arc<ProblemPattern>,
    n_jobs: usize,
    solver_threads: u32,
    workers: Vec<Mutex<Option<WorkerSolver>>>,
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
        let cones = cones_from_spec(&cone_kinds, cone_dims.as_slice()?)
            .map_err(PyValueError::new_err)?;
        let n_jobs = n_jobs.max(1);
        let workers = (0..n_jobs)
            .map(|_| Mutex::new(None))
            .collect();
        Ok(Self {
            pattern: Arc::new(ProblemPattern {
                n_vars,
                n_cons,
                p_indptr: i64_to_usize(p_indptr.as_slice()?, "p_indptr")?,
                p_indices: i64_to_usize(p_indices.as_slice()?, "p_indices")?,
                a_indptr: i64_to_usize(a_indptr.as_slice()?, "a_indptr")?,
                a_indices: i64_to_usize(a_indices.as_slice()?, "a_indices")?,
                cones,
                weight_start,
                weight_len,
            }),
            n_jobs,
            solver_threads: solver_threads.max(1),
            workers,
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
        let (n, p_stride) = contig_dims(p_nz.as_array(), "P")?;
        let (n_q, q_stride) = contig_dims(q.as_array(), "q")?;
        let (n_a, a_stride) = contig_dims(a_nz.as_array(), "A")?;
        let (n_b, b_stride) = contig_dims(b.as_array(), "b")?;
        let q_shared = shared_or_batch(n, n_q, "q")?;
        let a_shared = shared_or_batch(n, n_a, "A")?;
        let b_shared = shared_or_batch(n, n_b, "b")?;
        let p_rows = p_nz.as_slice()?;
        let q_rows = q.as_slice()?;
        let a_rows = a_nz.as_slice()?;
        let b_rows = b.as_slice()?;
        let pattern = Arc::clone(&self.pattern);
        let n_jobs = self.n_jobs;
        let solver_threads = self.solver_threads;

        let records = py
            .allow_threads(|| -> Result<_, String> {
                if n_jobs <= 1 || n <= 1 {
                    let mut guard = self.workers[0]
                        .lock()
                        .map_err(|err| format!("solver mutex poisoned: {err}"))?;
                    if guard.is_none() {
                        *guard = Some(WorkerSolver::new(Arc::clone(&pattern), solver_threads));
                    }
                    let worker = guard.as_mut().unwrap();
                    let mut out = Vec::with_capacity(n);
                    for i in 0..n {
                        let ps = i * p_stride;
                        let qs = if q_shared { 0 } else { i * q_stride };
                        let a_s = if a_shared { 0 } else { i * a_stride };
                        let bs = if b_shared { 0 } else { i * b_stride };
                        out.push(worker.solve_instance(
                            &p_rows[ps..ps + p_stride],
                            &q_rows[qs..qs + q_stride],
                            &a_rows[a_s..a_s + a_stride],
                            &b_rows[bs..bs + b_stride],
                            !q_shared || i == 0,
                            !a_shared || i == 0,
                        )?);
                    }
                    Ok(out)
                } else {
                    solve_batch(
                        &pattern,
                        &self.workers,
                        p_rows,
                        p_stride,
                        q_rows,
                        q_stride,
                        q_shared,
                        a_rows,
                        a_stride,
                        a_shared,
                        b_rows,
                        b_stride,
                        b_shared,
                        n,
                        solver_threads,
                    )
                }
            })
            .map_err(PyValueError::new_err)?;

        let n_rec = records.len();
        let wlen = self.pattern.weight_len.max(1);
        let mut ws = vec![0.0; n_rec * wlen];
        let mut objs = vec![0.0; n_rec];
        let mut iters = vec![0_i64; n_rec];
        let mut times = vec![0.0; n_rec];
        let mut statuses = Vec::with_capacity(n_rec);
        for (i, rec) in records.iter().enumerate() {
            let len = rec.x.len().min(wlen);
            ws[i * wlen..i * wlen + len].copy_from_slice(&rec.x[..len]);
            objs[i] = rec.obj;
            iters[i] = rec.iterations;
            times[i] = rec.solve_time;
            statuses.push(rec.status.clone());
        }

        let ws_arr = Array2::from_shape_vec((n_rec, wlen), ws)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let xs_arr = PyArray2::from_owned_array_bound(py, ws_arr);
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
