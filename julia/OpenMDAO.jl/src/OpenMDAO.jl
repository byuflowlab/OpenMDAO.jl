module OpenMDAO

using Pkg: Pkg
using OpenMDAOCore: OpenMDAOCore
using PythonCall: PythonCall

export om, make_component, DymosifiedCompWrapper

# load python api
const om = PythonCall.pynew()
const omjlcomps = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(om, PythonCall.pyimport("openmdao.api"))
    Pkg.add("OpenMDAOCore")
    PythonCall.pycopy!(omjlcomps, PythonCall.pyimport("omjlcomps"))
end

"""
    make_component(comp::OpenMDAOCore.AbstractComp)

Convinience method for creating either a `JuliaExplicitComp` or `JuliaImplicitComp`, depending on if `comp` is `<:OpenMDAOCore.AbstractExplicitComp` or `<:OpenMDAOCore.AbstractImplicitComp`, respectively.
"""
make_component
make_component(comp::OpenMDAOCore.AbstractExplicitComp) = omjlcomps.JuliaExplicitComp(jlcomp=comp)
make_component(comp::OpenMDAOCore.AbstractImplicitComp) = omjlcomps.JuliaImplicitComp(jlcomp=comp)

"""
    DymosifiedCompWrapper{Tkwargs}

Wrapper type that stores a type (not an instance) of an `<:OpenMDAOCore.AbstractComp` and the non-`num_nodes` keyword arguments (if any) used to construct the instance.

# Example
```julia-repl
julia> using OpenMDAOCore: OpenMDAOCore

julia> using OpenMDAO

julia> struct FooODE <: OpenMDAOCore.AbstractExplicitComp
         num_nodes::Int
         a::Float64
       end

julia> FooODE(; num_nodes, a) = FooODE(num_nodes, a)
FooODE

julia> dcw = OpenMDAO.DymosifiedCompWrapper(FooODE; a=8.0)
OpenMDAO.DymosifiedCompWrapper{Base.Pairs{Symbol, Float64, Tuple{Symbol}, NamedTuple{(:a,), Tuple{Float64}}}}(FooODE, Base.Pairs(:a => 8.0))

julia> comp = dcw(num_nodes=4)
Python JuliaExplicitComp: <omjlcomps.JuliaExplicitComp object at 0x7f38429333a0>

julia> 
```
"""
struct DymosifiedCompWrapper{Tkwargs}
    TComp::Type{<:OpenMDAOCore.AbstractComp}
    kwargs::Tkwargs
end

"""
    DymosifiedCompWrapper(TComp::Type{<:OpenMDAOCore.AbstractComp}; kwargs...)

Construct a wrapper to a `<:OpenMDAOCore.AbstractComp` that can be used to create a `omjlcomps.JuliaExplicitComp` or `omjlcomps.JuliaImplicitComp` by just passing a `num_nodes` argument.

# Arguments
* `TComp`: An `OpenMDAOCore.AbstractComp` type, not an instance (so `TComp = FooComp`, not `TComp = FooComp()`, analogous to the difference between `Float64` and `1.0`),
* `kwargs`: keyword arguments, other than `num_nodes`, that are needed to construct an instance of `TComp`

# Example
```julia-repl
julia> using OpenMDAOCore: OpenMDAOCore

julia> using OpenMDAO

julia> struct FooODE <: OpenMDAOCore.AbstractExplicitComp
         num_nodes::Int
         a::Float64
       end

julia> FooODE(; num_nodes, a) = FooODE(num_nodes, a)
FooODE

julia> dcw = OpenMDAO.DymosifiedCompWrapper(FooODE; a=8.0)
OpenMDAO.DymosifiedCompWrapper{Base.Pairs{Symbol, Float64, Tuple{Symbol}, NamedTuple{(:a,), Tuple{Float64}}}}(FooODE, Base.Pairs(:a => 8.0))

julia> comp = dcw(num_nodes=4)
Python JuliaExplicitComp: <omjlcomps.JuliaExplicitComp object at 0x7f38429333a0>

julia> 
```
"""
DymosifiedCompWrapper(TComp; kwargs...) = DymosifiedCompWrapper(TComp, kwargs)

function (dc::DymosifiedCompWrapper)(; num_nodes::Integer)
    return make_component(dc.TComp(; num_nodes, dc.kwargs...))
end

Base.length(::DymosifiedCompWrapper) = 1

end # module
