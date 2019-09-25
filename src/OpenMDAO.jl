module OpenMDAO
using PyCall
import Base.convert

export VarData, PartialsData, make_component

abstract type AbstractComp end
abstract type AbstractExplicitComp <: AbstractComp end
abstract type AbstractImplicitComp <: AbstractComp end

# Needed to avoid "Don't know how to convert PyObject to <some type> errors."
function convert(::Type{T}, po::PyObject) where {T<:AbstractComp}
    # Explaination of the difference between fields and properties:
    # https://discourse.julialang.org/t/whats-the-difference-between-fields-and-properties/12495
    args = [getproperty(po, n) for n in fieldnames(T)]
    return T(args...)
end


function setup(self::UnionAll)
    error("called dummy base setup with self{$(typeof(self))}")
end

function compute!(self::UnionAll, inputs, outputs)
    error("called dummy base compute! with self{$(typeof(self))}")
end

function compute_partials!(self::UnionAll, inputs, partials)
    error("called dummy base compute_partials! with self{$(typeof(self))}")
end

function apply_nonlinear!(self::UnionAll, inputs, outputs, residuals)
    error("called dummy base compute_partials! with self{$(typeof(self))}")
end

function linearize!(self::UnionAll, inputs, outputs, partials)
    error("called dummy base linearize! with self{$(typeof(self))}")
end

function guess_nonlinear!(self::UnionAll, inputs, outputs, residuals)
    error("called dummy base guess_nonlinear! with self{$(typeof(self))}")
end

function solve_nonlinear!(self::UnionAll, inputs, outputs)
    error("called dummy base guess_nonlinear! with self{$(typeof(self))}")
end

struct VarData
    name
    shape
    val
    units
end

VarData(name, shape, val; units=nothing) = VarData(name, shape, val, units)

struct PartialsData
    of
    wrt
    rows
    cols
    val
end

PartialsData(of, wrt; rows=nothing, cols=nothing, val=nothing) = PartialsData(of, wrt, rows, cols, val)

function get_pysetup(self::T) where {T}
    args = (T,)  # self
    ret = Tuple{Vector{VarData},  # input metadata
                Vector{VarData},  # output metadata
                Vector{PartialsData}}  # partials metadata
    return pyfunctionret(setup, ret, args...)
end

function get_pycompute(self::T) where {T}
    args = (T,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray})  # outputs

    return pyfunction(compute!, args...)
end

function get_pycompute_partials(self::T) where {T}
    # Initialize pycompute_partials to a dummy value.
    local pycompute_partials = nothing

    # Check for the existance of a compute_partials! method for the `self` type.
    args = (T,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{Tuple{String, String}, PyArray})  # partials
    try
        # Create the Python wrapper of the compute_partials! function.
        method = which(compute_partials!, args)
        pycompute_partials = pyfunction(compute_partials!, args...)
    catch err
        @warn "No compute_partials! method found for $(T)" 
    end

    return pycompute_partials
end

function get_pyapply_nonlinear(self::T) where {T}
    args = (T,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray},  # outputs
            PyDict{String, PyArray})  # residuals
    return pyfunction(apply_nonlinear!, args...)
end

function get_pylinearize(self::T) where {T}
    # Initialize pylinearize to a dummy value.
    local pylinearize = nothing
    args = (T,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray},  # outputs
            PyDict{Tuple{String, String}, PyArray})  # partials

    try
        # Create the Python wrapper of the linearize! function.
        method = which(linearize!, args)
        pylinearize = pyfunction(linearize!, args...)
    catch err
        @warn "No linearize! method found for $(T)" 
    end
    return pylinearize
end

function get_pyguess_nonlinear(self::T) where {T}
    # Initialize pyguess_nonlinear to a dummy value.
    local pyguess_nonlinear = nothing
    args = (T,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray},  # outputs
            PyDict{String, PyArray})  # residuals
    try
        # Create the Python wrapper of the guess_nonlinear! function.
        method = which(guess_nonlinear!, args)
        pyguess_nonlinear = pyfunction(guess_nonlinear!, args...)
    catch err
        @warn "No guess_nonlinear! method found for $(T)" 
    end

    return pyguess_nonlinear
end

function get_pysolve_nonlinear(self::T) where {T}
    # Initialize pyguess_nonlinear to a dummy value.
    local pysolve_nonlinear = nothing
    args = (T,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray})  # outputs
    try
        # Create the Python wrapper of the guess_nonlinear! function.
        method = which(solve_nonlinear!, args)
        pysolve_nonlinear = pyfunction(solve_nonlinear!, args...)
    catch err
        @warn "No solve_nonlinear! method found for $(T)" 
    end

    return pysolve_nonlinear
end

function make_component(self::T where {T<:AbstractExplicitComp})
    julia_comps = pyimport("omjl.julia_comps")
    comp = julia_comps.JuliaExplicitComp(julia_comp_data=self)
    return comp
end

function make_component(self::T where {T<:AbstractImplicitComp})
    julia_comps = pyimport("omjl.julia_comps")
    comp = julia_comps.JuliaImplicitComp(julia_comp_data=self)
    return comp
end

end # module
