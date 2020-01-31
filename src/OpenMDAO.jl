module OpenMDAO
using PyCall

include("utils.jl")

export VarData, PartialsData, make_component, AbstractExplicitComp, AbstractImplicitComp
export om  # direct access to Python module: openmdao.api

# load python api
const om = PyNULL()
const julia_comps = PyNULL()

function __init__()
    copy!(om, pyimport("openmdao.api"))
    copy!(julia_comps, pyimport("omjl.julia_comps"))
end

abstract type AbstractComp end
abstract type AbstractExplicitComp <: AbstractComp end
abstract type AbstractImplicitComp <: AbstractComp end

# The component_registry is a dict that stores each OpenMDAO.jl component struct
# that is created with the make_component methods. The idea is to avoid having
# to pass the <:AbstractComp structs from Python to Julia, because that involves
# copying the data, which is slow when the struct is large.
const CompIdType = BigInt
component_registry = Dict{CompIdType, AbstractComp}()

function make_component(self::T) where {T<:AbstractExplicitComp}
    comp_id = BigInt(objectid(self))
    component_registry[comp_id] = self
    comp = julia_comps.JuliaExplicitComp(jl_id=comp_id)
    return comp
end

function make_component(self::T) where {T<:AbstractImplicitComp}
    comp_id = BigInt(objectid(self))
    component_registry[comp_id] = self
    comp = julia_comps.JuliaImplicitComp(jl_id=comp_id)
    return comp
end

detect_compute_partials(::Type{<:AbstractExplicitComp}) = true
detect_guess_nonlinear(::Type{<:AbstractImplicitComp}) = true
detect_linearize(::Type{<:AbstractImplicitComp}) = true
detect_solve_nonlinear(::Type{<:AbstractImplicitComp}) = true

function setup(comp_id::Integer)
    comp = component_registry[comp_id]
    return setup(comp)
end

function setup(self::AbstractComp)
    error("called dummy base setup with self::$(typeof(self))")
end

function compute!(comp_id::Integer, inputs, outputs)
    comp = component_registry[comp_id]
    compute!(comp, inputs, outputs)
end

function compute!(self::AbstractComp, inputs, outputs)
    error("called dummy base compute! with self::$(typeof(self))")
end

function compute_partials!(comp_id::Integer, inputs, partials)
    comp = component_registry[comp_id]
    compute_partials!(comp, inputs, partials)
end

function compute_partials!(self::AbstractComp, inputs, partials)
    @warn "called dummy compute_partials! with self::$(typeof(self))"
end

function apply_nonlinear!(comp_id::Integer, inputs, outputs, residuals)
    comp = component_registry[comp_id]
    apply_nonlinear!(comp, inputs, outputs, residuals)
end

function apply_nonlinear!(self::AbstractComp, inputs, outputs, residuals)
    error("called dummy base compute_partials! with self::$(typeof(self))")
end

function linearize!(comp_id::Integer, inputs, outputs, partials)
    comp = component_registry[comp_id]
    linearize!(comp, inputs, outputs, partials)
end

function linearize!(self::AbstractComp, inputs, outputs, partials)
    error("called dummy base linearize! with self::$(typeof(self))")
end

function guess_nonlinear!(comp_id::Integer, inputs, outputs, residuals)
    comp = component_registry[comp_id]
    guess_nonlinear!(comp, inputs, outputs, residuals)
end

function guess_nonlinear!(self::AbstractComp, inputs, outputs, residuals)
    error("called dummy base guess_nonlinear! with self::$(typeof(self))")
end

function solve_nonlinear!(comp_id::Integer, inputs, outputs)
    comp = component_registry[comp_id]
    solve_nonlinear!(comp, inputs, outputs)
end

function solve_nonlinear!(self::AbstractComp, inputs, outputs)
    error("called dummy base guess_nonlinear! with self::$(typeof(self))")
end

struct VarData
    name
    val
    shape
    units
end

VarData(name; val=1.0, shape=1, units=nothing) = VarData(name, val, shape, units)

struct PartialsData
    of
    wrt
    rows
    cols
    val
end

PartialsData(of, wrt; rows=nothing, cols=nothing, val=nothing) = PartialsData(of, wrt, rows, cols, val)

function get_py2jl_setup(comp_id::Integer)
    args = (Integer,)  # self
    ret = Tuple{Vector{VarData},  # input metadata
                Vector{VarData},  # output metadata
                Vector{PartialsData}}  # partials metadata
    return pyfunctionret(setup, ret, args...)
end

function get_py2jl_compute(comp_id::Integer)
    args = (Integer,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray})  # outputs

    return pyfunction(compute!, args...)
end

function get_py2jl_compute_partials(comp_id::Integer)
    comp = component_registry[comp_id]
    T = typeof(comp)
    if detect_compute_partials(T)
        try
            # Look for the Python wrapper of the compute_partials! function.
            method = which(compute_partials!, (T,  # self
                                               PyDict{String, PyArray},  # inputs
                                               PyDict{Tuple{String, String}, PyArray}))  # partials)

            # Create a wrapper for the compute_partials! function.
            args = (Integer,  # component ID
                    PyDict{String, PyArray},  # inputs
                    PyDict{Tuple{String, String}, PyArray})  # partials
            pycompute_partials = pyfunction(compute_partials!, args...)
            return pycompute_partials
        catch err
            @warn "No compute_partials! method found for $(T)" 
            return nothing
        end
    else
        return nothing
    end
end

function get_py2jl_apply_nonlinear(comp_id::Integer)
    args = (Integer,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray},  # outputs
            PyDict{String, PyArray})  # residuals
    return pyfunction(apply_nonlinear!, args...)
end

function get_py2jl_linearize(comp_id::Integer)
    comp = component_registry[comp_id]
    T = typeof(comp)
    if detect_linearize(T)
        try
            method = which(linearize!, (T, 
                                        PyDict{String, PyArray}, # inputs
                                        PyDict{String, PyArray}, # outputs
                                        PyDict{Tuple{String, String}, PyArray})) # partials
            args = (Integer, 
                    PyDict{String, PyArray}, # inputs
                    PyDict{String, PyArray}, # outputs
                    PyDict{Tuple{String, String}, PyArray})
            # Create the Python wrapper of the linearize! function.
            pylinearize = pyfunction(linearize!, args...)
            return pylinearize
        catch err
            @warn "No linearize! method found for $(T)" 
            return nothing
        end
    else
        return nothing
    end
end

function get_py2jl_guess_nonlinear(comp_id::Integer)
    comp = component_registry[comp_id]
    T = typeof(comp)
    if detect_guess_nonlinear(T)
        try
            # check if the real function exists
            method = which(guess_nonlinear!, (T, 
                                              PyDict{String, PyArray}, 
                                              PyDict{String, PyArray}))

            # if it does, then return the integer id version
            args = (Integer,  # self
                    PyDict{String, PyArray},  # inputs
                    PyDict{String, PyArray})  # outputs
            pyguess_nonlinear = pyfunction(guess_nonlinear!, args...)
            return pyguess_nonlinear
        catch err
            @warn "No guess_nonlinear! method found for $(T)" 
            return nothing
        end
    else
        return nothing
    end
end

function get_py2jl_solve_nonlinear(comp_id::Integer)
    comp = component_registry[comp_id]
    T = typeof(comp)
    if detect_solve_nonlinear(T)
        try
            method = which(solve_nonlinear!, (T, 
                                              PyDict{String, PyArray}, 
                                              PyDict{String, PyArray}))
            args = (Integer,  # self
                    PyDict{String, PyArray},  # inputs
                    PyDict{String, PyArray})  # outputs
            pysolve_nonlinear = pyfunction(solve_nonlinear!, args...)
            return pysolve_nonlinear

        catch err
            @warn "No solve_nonlinear! method found for $(T)" 
            return nothing
        end
    else
        return nothing
    end
end

# include("example_components/simple_explicit.jl")
# include("example_components/simple_implicit.jl")
# include("example_components/actuator_disc.jl")

end # module
