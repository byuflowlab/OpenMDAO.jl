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
detect_linearize(::Type{<:AbstractImplicitComp}) = true
detect_apply_nonlinear(::Type{<:AbstractImplicitComp}) = true
detect_guess_nonlinear(::Type{<:AbstractImplicitComp}) = true
detect_solve_nonlinear(::Type{<:AbstractImplicitComp}) = true
detect_apply_linear(::Type{<:AbstractImplicitComp}) = true

function setup(comp_id::Integer)
    comp = component_registry[comp_id]
    return setup(comp)
end

function compute!(comp_id::Integer, inputs, outputs)
    comp = component_registry[comp_id]
    compute!(comp, inputs, outputs)
end

function compute_partials!(comp_id::Integer, inputs, partials)
    comp = component_registry[comp_id]
    compute_partials!(comp, inputs, partials)
end

function apply_nonlinear!(comp_id::Integer, inputs, outputs, residuals)
    comp = component_registry[comp_id]
    apply_nonlinear!(comp, inputs, outputs, residuals)
end

function linearize!(comp_id::Integer, inputs, outputs, partials)
    comp = component_registry[comp_id]
    linearize!(comp, inputs, outputs, partials)
end

function guess_nonlinear!(comp_id::Integer, inputs, outputs, residuals)
    comp = component_registry[comp_id]
    guess_nonlinear!(comp, inputs, outputs, residuals)
end

function solve_nonlinear!(comp_id::Integer, inputs, outputs)
    comp = component_registry[comp_id]
    solve_nonlinear!(comp, inputs, outputs)
end

function apply_linear!(comp_id::Integer, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    comp = component_registry[comp_id]
    apply_linear!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
end

function setup(self::AbstractComp)
    error("called dummy base setup with self::$(typeof(self))")
end

function compute!(self::AbstractExplicitComp, inputs, outputs)
    error("called dummy base compute! with self::$(typeof(self))")
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
            # Look for the method for type T.
            method = which(compute_partials!, (T,  # self
                                               PyDict{String, PyArray},  # inputs
                                               PyDict{Tuple{String, String}, PyArray}))  # partials)

            # Create a Python wrapper for the method.
            args = (Integer,  # component ID
                    PyDict{String, PyArray},  # inputs
                    PyDict{Tuple{String, String}, PyArray})  # partials
            return pyfunction(compute_partials!, args...)
        catch err
            @warn "No compute_partials! method found for $(T)" 
            return nothing
        end
    else
        return nothing
    end
end

function get_py2jl_apply_nonlinear(comp_id::Integer)
    comp = component_registry[comp_id]
    T = typeof(comp)
    if detect_apply_nonlinear(T)
        try
            # Look for the method for type T.
            method = which(apply_nonlinear!, (T,
                                              PyDict{String, PyArray},
                                              PyDict{String, PyArray},
                                              PyDict{String, PyArray}))

            # Create a Python wrapper for the method.
            args = (Integer,  # self
                    PyDict{String, PyArray},  # inputs
                    PyDict{String, PyArray},  # outputs
                    PyDict{String, PyArray})  # residuals
            return pyfunction(apply_nonlinear!, args...)
        catch err
            @warn "No apply_nonlinear! method found for $(T)" 
            return nothing
        end
    else
        return nothing
    end
end

function get_py2jl_linearize(comp_id::Integer)
    comp = component_registry[comp_id]
    T = typeof(comp)
    if detect_linearize(T)
        try
            # Look for the method for type T.
            method = which(linearize!, (T, 
                                        PyDict{String, PyArray}, # inputs
                                        PyDict{String, PyArray}, # outputs
                                        PyDict{Tuple{String, String}, PyArray})) # partials

            # Create a Python wrapper for the method.
            args = (Integer, 
                    PyDict{String, PyArray}, # inputs
                    PyDict{String, PyArray}, # outputs
                    PyDict{Tuple{String, String}, PyArray})
            return pyfunction(linearize!, args...)
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
            # Look for the method for type T.
            method = which(guess_nonlinear!, (T, 
                                              PyDict{String, PyArray}, 
                                              PyDict{String, PyArray},
                                              PyDict{String, PyArray}))

            # Create a Python wrapper for the method.
            args = (Integer,  # self
                    PyDict{String, PyArray},  # inputs
                    PyDict{String, PyArray},  # outputs
                    PyDict{String, PyArray})  # residuals
            return pyfunction(guess_nonlinear!, args...)
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
            # Look for the method for type T.
            method = which(solve_nonlinear!, (T, 
                                              PyDict{String, PyArray}, 
                                              PyDict{String, PyArray}))

            # Create a Python wrapper for the method.
            args = (Integer,  # self
                    PyDict{String, PyArray},  # inputs
                    PyDict{String, PyArray})  # outputs
            return pyfunction(solve_nonlinear!, args...)
        catch err
            @warn "No solve_nonlinear! method found for $(T)" 
            return nothing
        end
    else
        return nothing
    end
end

function get_py2jl_apply_linear(comp_id::Integer)
    comp = component_registry[comp_id]
    T = typeof(comp)
    if detect_apply_linear(T)
        try
            # Look for the method for type T.
            method = which(apply_linear!, (T, 
                                           PyDict{String, PyArray}, 
                                           PyDict{String, PyArray},
                                           PyDict{String, PyArray},
                                           PyDict{String, PyArray},
                                           PyDict{String, PyArray},
                                           String))

            # Create a Python wrapper for the method.
            args = (Integer,  # self
                    PyDict{String, PyArray},  # inputs
                    PyDict{String, PyArray},  # outputs
                    PyDict{String, PyArray},  # d_inputs
                    PyDict{String, PyArray},  # d_outputs
                    PyDict{String, PyArray},  # d_residuals
                    String)                   # mode
            return pyfunction(apply_linear!, args...)
        catch err
            @warn "No apply_linear! method found for $(T)" 
            return nothing
        end
    else
        return nothing
    end
end

end # module
