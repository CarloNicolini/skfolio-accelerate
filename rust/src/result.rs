//! Native solve records returned to Python.

#[derive(Clone, Debug)]
pub struct SolveRecord {
    pub x: Vec<f64>,
    pub status: String,
    pub obj: f64,
    pub iterations: i64,
    pub solve_time: f64,
}
