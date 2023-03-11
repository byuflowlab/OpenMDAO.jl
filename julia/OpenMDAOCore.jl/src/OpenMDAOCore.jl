module OpenMDAOCore

include("utils.jl")

export VarData, PartialsData, AbstractExplicitComp, AbstractImplicitComp
export has_compute_partials, has_compute_jacvec_product
export has_apply_nonlinear, has_solve_nonlinear, has_linearize, has_apply_linear, has_solve_linear, has_guess_nonlinear 

abstract type AbstractComp end
abstract type AbstractExplicitComp <: AbstractComp end
abstract type AbstractImplicitComp <: AbstractComp end

struct OpenMDAOMethodError{T} <: Exception
    method_name::String
end

function OpenMDAOMethodError(self::AbstractComp, method_name)
    T = typeof(self)
    return OpenMDAOMethodError{T}(method_name)
end

function Base.showerror(io::IO, e::OpenMDAOMethodError{T}) where {T}
    print(io, "called fallback $(e.method_name) method for type $T")
end

function setup(self::AbstractComp)
    throw(OpenMDAOMethodError(self, "setup"))
    return nothing
end

function compute!(self::AbstractExplicitComp, inputs, outputs)
    throw(OpenMDAOMethodError(self, "compute!"))
    return nothing
end

function compute_partials!(self::AbstractExplicitComp, inputs, partials)
    throw(OpenMDAOMethodError(self, "compute_partials!"))
    return nothing
end

function compute_jacvec_product!(self::AbstractExplicitComp, inputs, d_inputs, d_outputs, mode)
    throw(OpenMDAOMethodError(self, "compute_jacvec_product!"))
    return nothing
end

function apply_nonlinear!(self::AbstractImplicitComp, inputs, outputs, residuals)
    throw(OpenMDAOMethodError(self, "apply_nonlinear!"))
    return nothing
end

function solve_nonlinear!(self::AbstractImplicitComp, inputs, outputs)
    throw(OpenMDAOMethodError(self, "solve_nonlinear!"))
    return nothing
end

function linearize!(self::AbstractImplicitComp, inputs, outputs, partials)
    throw(OpenMDAOMethodError(self, "linearize!"))
    return nothing
end

function apply_linear!(self::AbstractImplicitComp, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    throw(OpenMDAOMethodError(self, "apply_linear!"))
    return nothing
end

function solve_linear!(self::AbstractImplicitComp, d_outputs, d_residuals, mode)
    throw(OpenMDAOMethodError(self, "solve_linear!"))
    return nothing
end

function guess_nonlinear!(self::AbstractImplicitComp, inputs, outputs, residuals)
    throw(OpenMDAOMethodError(self, "guess_nonlinear!"))
    return nothing
end

function has_compute_partials(self::AbstractExplicitComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(compute_partials!, (T, Any, Any))
    # Next, get the fallback method.
    fallback = which(compute_partials!, (AbstractExplicitComp, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
end

function has_compute_jacvec_product(self::AbstractExplicitComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(compute_jacvec_product!, (T, Any, Any, Any, Any))
    # Next, get the fallback method.
    fallback = which(compute_jacvec_product!, (AbstractExplicitComp, Any, Any, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
end

function has_apply_nonlinear(self::AbstractImplicitComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(apply_nonlinear!, (T, Any, Any, Any))
    # Next, get the fallback method.
    fallback = which(apply_nonlinear!, (AbstractImplicitComp, Any, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
end

function has_solve_nonlinear(self::AbstractImplicitComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(solve_nonlinear!, (T, Any, Any))
    # Next, get the fallback method.
    fallback = which(solve_nonlinear!, (AbstractImplicitComp, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
end

function has_linearize(self::AbstractImplicitComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(linearize!, (T, Any, Any, Any))
    # Next, get the fallback method.
    fallback = which(linearize!, (AbstractImplicitComp, Any, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
end

function has_apply_linear(self::AbstractImplicitComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(apply_linear!, (T, Any, Any, Any, Any, Any, Any))
    # Next, get the fallback method.
    fallback = which(apply_linear!, (AbstractImplicitComp, Any, Any, Any, Any, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
end

function has_solve_linear(self::AbstractImplicitComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(solve_linear!, (T, Any, Any, Any))
    # Next, get the fallback method.
    fallback = which(solve_linear!, (AbstractImplicitComp, Any, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
end

function has_guess_nonlinear(self::AbstractImplicitComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(guess_nonlinear!, (T, Any, Any, Any))
    # Next, get the fallback method.
    fallback = which(guess_nonlinear!, (AbstractImplicitComp, Any, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
end

_vd_checkbounds(var::Float64, shape) = true
_vd_checkbounds(var::Nothing, shape) = true
_vd_checkbounds(var::AbstractArray, shape) = size(var) == shape


"""
    VarData(name::String; <keyword arguments>)

Create a VarData object for an OpenMDAO variable named `name`.

`VarData` objects are used to construct arguments to OpenMDAO's `Component.add_input` and `Component.add_output` methods.
Specifically, if a `VarData` object `var` refers to an input variable, the `Component.add_input` call will look like this:

```python
Component.add_input(var.name, shape=var.shape, val=var.val, units=var.units, tags=var.tags)
```

and if the `VarData` object `var` is an output variable, the `Component.add_output` call will look like this:

```python
Component.add_output(var.name, shape=var.shape, val=var.val, units=var.units, lower=var.lower, upper=var.upper, tags=var.tags)
```

The `name` positional argument is required and sets the `name` field.
The value of the other `VarData` fields (e.g., `var.shape`, `var.val`, etc.) are set via constructor keyword arguments, here:

# Keyword Arguments
- `val::Union{Float64,<:AbstractArray{Float64},Nothing} = 1.0`: variable's default value, set to `1.0` if `nothing`.
- `shape::Union{Int64,NTuple{N,Int64},Nothing} = (1,)`: variable shape, set to `(1,)` if `nothing`.
- `units::Union{String,Nothing} = nothing`: variable units.
- `lower::Union{Float64,<:AbstractArray{Float64,N},Nothing} = nothing`: variable's lower limit.
- `upper::Union{Float64,<:AbstractArray{Float64,N},Nothing} = nothing`: variable's upper limit.
- `tags::Union{<:AbstractVector{String},Nothing} = nothing`: variable tags.
"""
struct VarData{N,TVal<:Union{Float64,<:AbstractArray{Float64,N}},TUnits<:Union{String,Nothing},TBounds<:Union{Float64,<:AbstractArray{Float64,N}, Nothing}, TTags<:Union{<:AbstractVector{String},Nothing}}
    name::String
    val::TVal
    shape::NTuple{N,Int64}
    units::TUnits
    lower::TBounds
    upper::TBounds
    tags::TTags

    function VarData(name, val::TVal, shape::NTuple{N,Int64}, units, lower::TBounds, upper::TBounds, tags::TTags=nothing) where {N,TVal<:Union{Float64,<:AbstractArray{Float64,N}},TBounds,TTags}
        _vd_checkbounds(val, shape) || throw(ArgumentError("size of val argument $(size(val)) should match shape argument $(shape)"))
        _vd_checkbounds(lower, shape) || throw(ArgumentError("size of lower argument $(size(lower)) should match shape argument $(shape)"))
        _vd_checkbounds(upper, shape) || throw(ArgumentError("size of upper argument $(size(upper)) should match shape argument $(shape)"))
        return new{N,typeof(val),typeof(units),TBounds,TTags}(name, val, shape, units, lower, upper, tags)
    end
end

VarData(name; val=nothing, shape=nothing, units=nothing, lower=nothing, upper=nothing, tags=nothing) = VarData(name, val, shape, units, lower, upper, tags)

VarData(name, val::Float64, shape::Nothing, units, lower, upper, tags=nothing) = VarData(name, val, (1,), units, lower, upper, tags)
VarData(name, val::AbstractArray{Float64,N}, shape::Nothing, units, lower, upper, tags=nothing) where {N} = VarData(name, val, size(val), units, lower, upper, tags)
VarData(name, val::Nothing, shape::Nothing, units, lower, upper, tags=nothing) = VarData(name, 1.0, (1,), units, lower, upper, tags)

VarData(name, val::Float64, shape::Int64, units, lower, upper, tags=nothing) = VarData(name, val, (shape,), units, lower, upper, tags)
VarData(name, val::AbstractArray{Float64,N}, shape::Int64, units, lower, upper, tags=nothing) where {N} = VarData(name, val, (shape,), units, lower, upper, tags)
VarData(name, val::Nothing, shape::Int64, units, lower, upper, tags=nothing) = VarData(name, 1.0, (shape,), units, lower, upper, tags)

# These next two are commented out because they're handled by the `VarData` inner constructor.
# VarData(name, val::Float64, shape::NTuple{N,Int64}, units, lower, upper) where {N} = VarData(name, val, shape, units, lower, upper)
# VarData(name, val::AbstractArray{Float64,N}, shape::NTuple{N,Int64}, units, lower, upper) where {N} = VarData(name, val, shape, units, lower, upper)
VarData(name, val::Nothing, shape::NTuple{N,Int64}, units, lower, upper, tags=nothing) where {N} = VarData(name, 1.0, shape, units, lower, upper, tags)

_pd_rc_checkbounds(rows::Nothing, cols::Nothing) = true
_pd_rc_checkbounds(rows::AbstractVector, cols::Nothing) = false
_pd_rc_checkbounds(rows::Nothing, cols::AbstractVector) = false
_pd_rc_checkbounds(rows::AbstractVector, cols::AbstractVector) = size(rows) == size(cols)

_pd_val_checkbounds(val::Nothing, rows::Nothing) = true
_pd_val_checkbounds(val::AbstractArray, rows::Nothing) = true
_pd_val_checkbounds(val::Nothing, rows::AbstractVector) = true
_pd_val_checkbounds(val::AbstractVector, rows::AbstractVector) = size(val) == size(rows)
_pd_val_checkbounds(val::AbstractArray, rows::AbstractVector) = false
_pd_val_checkbounds(val::Float64, rows) = true

"""
    PartialsData(of::String, wrt::String; <keyword arguments>)

Create a PartialsData object for the derivative of variable `of` with respect to variable `wrt`.

`PartialsData` objects are used to construct arguments to OpenMDAO's `Component.declare_partials` method.
Specifically, a `PartialsData` object `pd` will eventually be used to call `Component.declare_partials` like this:

```python
Component.declare_partials(pd.of, pd.wrt, rows=pd.rows, cols=pd.cols, val=pd.val, method=pd.method)
```

The `of` and `wrt` positional arguments are required and set the `of` and `wrt` fields.
The value of the other `PartialsData` fields (e.g., `pd.rows`, `pd.val`, etc.) are set via constructor keyword arguments, here:

# Keyword Arguments
- `rows::Union{<:AbstractVector{Int64},Nothing} = nothing`: row indices for each non-zero Jacobian entry, if not `nothing`.
- `cols::Union{<:AbstractVector{Int64},Nothing} = nothing`: column indices for each non-zero Jacobian entry, if not `nothing`.
- `val::Union{Float64,<:AbstractArray{Float64},Nothing} = nothing`: value of Jacobian, if not `nothing`.
- `method::String = "exact"`: method use to calcluate the partial derivative(s). Should be one of

    * `"exact"`: user-defined partial derivatives via `compute_partials!`, `linearize!`, etc.
    * `"fd"`: finite difference approximation
    * `"cs"`: complex step approximation
"""
struct PartialsData{N,TRows<:Union{<:AbstractVector{Int64},Nothing},TCols<:Union{<:AbstractVector{Int64},Nothing},TVal<:Union{Float64,<:AbstractArray{Float64,N},Nothing}}
    of::String
    wrt::String
    rows::TRows
    cols::TCols
    val::TVal
    method::String

    function PartialsData(of, wrt, rows, cols, val::Union{Float64,Nothing,AbstractVector{Float64}}, method)
        _pd_rc_checkbounds(rows, cols) || throw(ArgumentError("size of rows argument $(rows) should match size of cols argument $(cols)"))
        _pd_val_checkbounds(val, rows) || throw(ArgumentError("size of val argument $(val) should match size of rows/cols argument $(rows)"))
        return new{1,typeof(rows),typeof(cols),typeof(val)}(of, wrt, rows, cols, val, method)
    end

    function PartialsData(of, wrt, rows, cols, val::AbstractArray{Float64, N}, method) where {N}
        _pd_rc_checkbounds(rows, cols) || throw(ArgumentError("size of rows argument $(rows) should match size of cols argument $(cols)"))
        _pd_val_checkbounds(val, rows) || throw(ArgumentError("size of val argument $(val) should match size of rows/cols argument $(rows)"))
        return new{N,typeof(rows),typeof(cols),typeof(val)}(of, wrt, rows, cols, val, method)
    end
end

PartialsData(of, wrt; rows=nothing, cols=nothing, val=nothing, method="exact") = PartialsData(of, wrt, rows, cols, val, method)

end # module
