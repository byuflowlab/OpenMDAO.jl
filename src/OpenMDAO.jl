module OpenMDAO
using PyCall
import Base.convert

# Importing the OpenMDAO Python module with pyimport and then exporting it makes
# om a "NULL PyObject." Seems like any modules imported with pyimport have to be
# used in the same scope they're imported in.
# om = pyimport("openmdao.api")
# export VarData, PartialsData, make_component, om

export VarData, PartialsData, make_component, AbstractExplicitComp, AbstractImplicitComp
export om  # direct access to Python module: openmdao.api

# load python api
const om = PyNULL()

function __init__()
    copy!(om, pyimport("openmdao.api"))
end




abstract type AbstractComp end
abstract type AbstractExplicitComp <: AbstractComp end
abstract type AbstractImplicitComp <: AbstractComp end


# Needed to avoid "Don't know how to convert PyObject to <some type> errors."
# function convert(::Type{T}, po::PyObject) where {T<:AbstractComp}
#     # Explaination of the difference between fields and properties:
#     # https://discourse.julialang.org/t/whats-the-difference-between-fields-and-properties/12495
#     args = [getproperty(po, n) for n in fieldnames(T)]
#     return T(args...)
# end


function setup(self::Int)
    comp = component_registry[self]
    return setup(comp)
end

function setup(self::AbstractComp)
    error("called dummy base setup with self{$(typeof(self))}")
end

function compute!(self::Int, inputs, outputs)
    comp = component_registry[self]
    compute!(comp, inputs, outputs)
end

function compute!(self::AbstractComp, inputs, outputs)
    error("called dummy base compute! with self{$(typeof(self))}")
end

function compute_partials!(self::Int, inputs, partials)
    comp = component_registry[self]
    compute_partials!(comp, inputs, partials)
end

function compute_partials!(self::AbstractComp, inputs, partials)
    error("called dummy base compute_partials! with self{$(typeof(self))}")
end

function apply_nonlinear!(self::Int, inputs, outputs, residuals)
    comp = component_registry[self]
    apply_nonlinear!(comp, inputs, outputs, residuals)
end

function apply_nonlinear!(self::AbstractComp, inputs, outputs, residuals)
    error("called dummy base compute_partials! with self{$(typeof(self))}")
end

function linearize!(self::Int, inputs, outputs, partials)
    comp = component_registry[self]
    linearize!(comp, inputs, outputs, partials)
end

function linearize!(self::AbstractComp, inputs, outputs, partials)
    error("called dummy base linearize! with self{$(typeof(self))}")
end

function guess_nonlinear!(self::Int, inputs, outputs, residuals)
    comp = component_registry[self]
    guess_nonlinear!(comp, inputs, outputs, residuals)
end

function guess_nonlinear!(self::AbstractComp, inputs, outputs, residuals)
    error("called dummy base guess_nonlinear! with self{$(typeof(self))}")
end

function solve_nonlinear!(self::Int, inputs, outputs)
    comp = component_registry[self]
    solve_nonlinear!(comp, inputs, outputs)
end

function solve_nonlinear!(self::AbstractComp, inputs, outputs)
    error("called dummy base guess_nonlinear! with self{$(typeof(self))}")
end

struct VarData
    name
    val
    shape
    units
end

# VarData(name, shape, val; units=nothing) = VarData(name, shape, val, units)
VarData(name; val=1.0, shape=1, units=nothing) = VarData(name, val, shape, units)

struct PartialsData
    of
    wrt
    rows
    cols
    val
end

PartialsData(of, wrt; rows=nothing, cols=nothing, val=nothing) = PartialsData(of, wrt, rows, cols, val)

function get_pysetup(self::Int)
    args = (Int,)  # self
    ret = Tuple{Vector{VarData},  # input metadata
                Vector{VarData},  # output metadata
                Vector{PartialsData}}  # partials metadata
    return pyfunctionret(setup, ret, args...)
end

function get_pycompute(self::Int)
    args = (Int,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray})  # outputs

    return pyfunction(compute!, args...)
end

function get_pycompute_partials(self::Int)
    try
        comp = component_registry[self]
        # Create the Python wrapper of the compute_partials! function.
        method = which(compute_partials!, (typeof(comp),  # self
                                           PyDict{String, PyArray},  # inputs
                                           PyDict{Tuple{String, String}, PyArray}))  # partials)

        args = (Int,  # self
                PyDict{String, PyArray},  # inputs
                PyDict{Tuple{String, String}, PyArray})  # partials
        pycompute_partials = pyfunction(compute_partials!, args...)
        return pycompute_partials
    catch err
        # @warn "No compute_partials! method found for $(T)" 
        return nothing
    end
end

function get_pyapply_nonlinear(self::Int)
    args = (Int,  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray},  # outputs
            PyDict{String, PyArray})  # residuals
    return pyfunction(apply_nonlinear!, args...)
end

function get_pylinearize(self::Int)
    try
        comp = component_registry[self]
        method = which(linearize!, (typeof(comp), 
                                    PyDict{String, PyArray}, # inputs
                                    PyDict{String, PyArray}, # outputs
                                    PyDict{Tuple{String, String}, PyArray})) # partials
        args = (Int, 
                PyDict{String, PyArray}, # inputs
                PyDict{String, PyArray}, # outputs
                PyDict{Tuple{String, String}, PyArray})
        # Create the Python wrapper of the linearize! function.
        pylinearize = pyfunction(linearize!, args...)
        return pylinearize
    catch err
        # @warn "No linearize! method found for $(T)" 
        return nothing
    end
end

function get_pyguess_nonlinear(self::Int)
    try
        comp = component_registry[self]
        # check if the real function exists
        method = which(guess_nonlinear!, (typeof(comp), 
                                          PyDict{String, PyArray}, 
                                          PyDict{String, PyArray}))

        # if it does, then return the integer id version
        args = (Int,  # self
                PyDict{String, PyArray},  # inputs
                PyDict{String, PyArray})  # outputs
        pyguess_nonlinear = pyfunction(guess_nonlinear!, args...)
        return pyguess_nonlinear
    catch err
        return nothing
    end
end

function get_pysolve_nonlinear(self::Int)

    try
        comp = component_registry[self]
        method = which(solve_nonlinear!, (typeof(comp), 
                                          PyDict{String, PyArray}, 
                                          PyDict{String, PyArray}))
        args = (Int,  # self
                PyDict{String, PyArray},  # inputs
                PyDict{String, PyArray})  # outputs
        pysolve_nonlinear = pyfunction(solve_nonlinear!, args...)
        return pysolve_nonlinear

    catch err
        return nothing
    end
end

component_registry = AbstractComp[]

function make_component(self::T where {T<:AbstractExplicitComp})
    julia_comps = pyimport("omjl.julia_comps")
    push!(component_registry, self)
    comp_id = length(component_registry)
    comp = julia_comps.JuliaExplicitComp(jl_id=comp_id)
    return comp
end

function make_component(self::T where {T<:AbstractImplicitComp})
    julia_comps = pyimport("omjl.julia_comps")
    push!(component_registry, self)
    comp_id = length(component_registry)
    comp = julia_comps.JuliaImplicitComp(jl_id=comp_id)
    return comp
end

end # module
