"""Persistent COSMO.jl workspace used by skfolio-accelerate.

The Python compact engines call ``solve_qp!`` once per fold. The first call
assembles the model. Later ``q``/``b``-only folds use public ``update!``
(factorization reuse). When ``P`` or ``A`` change, the model is reassembled
because Ruiz scaling makes in-place ``nzval`` writes illegal; ADMM iterates
are still warm-started.
"""
module CosmoBridge

using COSMO
using SparseArrays

export Workspace, make_workspace, solve_qp!, warmup!

mutable struct Workspace
    model::COSMO.Model{Float64}
    settings::COSMO.Settings{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    assembled::Bool
    n_iter::Int
    status::String
end

function default_settings(;
    eps_abs::Float64 = 1e-8,
    eps_rel::Float64 = 1e-8,
    max_iter::Int = 10000,
    empty_accelerator::Bool = false,
)
    settings = COSMO.Settings{Float64}()
    settings.verbose = false
    settings.eps_abs = eps_abs
    settings.eps_rel = eps_rel
    settings.max_iter = max_iter
    settings.scaling = 10
    settings.decompose = false
    if hasfield(typeof(settings), :compact_transformation)
        settings.compact_transformation = false
    end
    if empty_accelerator
        # Anderson acceleration cycles on poorly-scaled LPs (MAD) and can
        # block adaptive rho. EmptyAccelerator restores vanilla ADMM.
        settings.accelerator = COSMO.with_options(COSMO.EmptyAccelerator)
        settings.adaptive_rho_interval = 25
    end
    return settings
end

function make_workspace(;
    eps_abs::Float64 = 1e-8,
    eps_rel::Float64 = 1e-8,
    max_iter::Int = 10000,
    empty_accelerator::Bool = false,
)
    return Workspace(
        COSMO.Model{Float64}(),
        default_settings(;
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
            empty_accelerator=empty_accelerator,
        ),
        Float64[],
        Float64[],
        false,
        0,
        "Unsolved",
    )
end

function _as_int_vector(values)::Vector{Int}
    n = length(values)
    out = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        out[i] = Int(values[i])
    end
    return out
end

function _as_float_vector(values)::Vector{Float64}
    n = length(values)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = Float64(values[i])
    end
    return out
end

function _sparse(m::Int, n::Int, colptr, rowval, nzval)
    return SparseMatrixCSC{Float64,Int}(
        m,
        n,
        _as_int_vector(colptr),
        _as_int_vector(rowval),
        _as_float_vector(nzval),
    )
end

function make_sets(n_zero::Int, n_nonneg::Int, soc_dims, n_exp::Int, box_l, box_u)
    sets = COSMO.AbstractConvexSet{Float64}[]
    n_zero > 0 && push!(sets, COSMO.ZeroSet{Float64}(n_zero))
    n_nonneg > 0 && push!(sets, COSMO.Nonnegatives{Float64}(n_nonneg))
    for dim in soc_dims
        push!(sets, COSMO.SecondOrderCone{Float64}(Int(dim)))
    end
    for _ in 1:n_exp
        push!(sets, COSMO.ExponentialCone{Float64}())
    end
    if length(box_l) > 0
        push!(
            sets,
            COSMO.Box{Float64}(_as_float_vector(box_l), _as_float_vector(box_u)),
        )
    end
    return sets
end

function mark_kkt_dirty!(model::COSMO.Model)
    model.states.KKT_FACTORED = false
    model.kkt_solver = nothing
    return nothing
end

function cold_start!(model::COSMO.Model{Float64})
    fill!(model.vars.x, 0.0)
    fill!(model.vars.μ, 0.0)
    fill!(model.vars.s.data, 0.0)
    return nothing
end

function same_pattern(left::SparseMatrixCSC, right::SparseMatrixCSC)
    return size(left) == size(right) &&
           length(left.nzval) == length(right.nzval) &&
           length(left.colptr) == length(right.colptr)
end

function solve_qp!(
    ws::Workspace,
    n_cons::Int,
    n_vars::Int,
    P_colptr,
    P_rowval,
    P_nzval,
    q,
    A_colptr,
    A_rowval,
    A_nzval,
    b,
    n_zero::Int,
    n_nonneg::Int,
    soc_dims,
    n_exp::Int,
    box_l,
    box_u,
    warm::Bool,
    update_p::Bool,
    update_a::Bool,
)
    P = _sparse(n_vars, n_vars, P_colptr, P_rowval, P_nzval)
    A = _sparse(n_cons, n_vars, A_colptr, A_rowval, A_nzval)
    q_vec = _as_float_vector(q)
    b_vec = _as_float_vector(b)
    sets = make_sets(n_zero, n_nonneg, soc_dims, n_exp, box_l, box_u)

    if !ws.assembled
        COSMO.set!(ws.model, P, q_vec, A, b_vec, sets, ws.settings)
        ws.assembled = true
        ws.x = zeros(Float64, n_vars)
        ws.y = zeros(Float64, n_cons)
    elseif !update_p && !update_a && same_pattern(ws.model.p.P, P) &&
           same_pattern(ws.model.p.A, A)
        # Public API: q/b updates reuse the KKT factorisation.
        COSMO.update!(ws.model; q = q_vec, b = b_vec)
        if warm && length(ws.x) == n_vars && length(ws.y) == n_cons
            COSMO.warm_start!(ws.model, ws.x, ws.y)
        else
            cold_start!(ws.model)
        end
    else
        # P or A changed: reassemble (scaling makes in-place nzval writes unsafe)
        # but keep ADMM iterates for the next optimize!.
        x0 = copy(ws.x)
        y0 = copy(ws.y)
        COSMO.empty_model!(ws.model)
        COSMO.set!(ws.model, P, q_vec, A, b_vec, sets, ws.settings)
        if warm && length(x0) == n_vars && length(y0) == n_cons
            COSMO.warm_start!(ws.model, x0, y0)
        end
        if length(ws.x) != n_vars
            ws.x = zeros(Float64, n_vars)
        end
        if length(ws.y) != n_cons
            ws.y = zeros(Float64, n_cons)
        end
    end

    result = COSMO.optimize!(ws.model)
    copyto!(ws.x, result.x)
    copyto!(ws.y, result.y)
    ws.n_iter = Int(result.iter)
    ws.status = string(result.status)
    return ws.x
end

function warmup!()
    ws = make_workspace()
    solve_qp!(
        ws,
        1,
        1,
        [1, 2],
        [1],
        [1.0],
        [0.0],
        [1, 2],
        [1],
        [1.0],
        [1.0],
        1,
        0,
        Int[],
        0,
        Float64[],
        Float64[],
        false,
        false,
        false,
    )
    return nothing
end

end # module
