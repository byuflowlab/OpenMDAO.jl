_vd_checkbounds(var::Float64, shape) = true
_vd_checkbounds(var::Nothing, shape) = true
_vd_checkbounds(var::AbstractArray, shape) = size(var) == shape


"""
    VarData(name::String; <keyword arguments>)

Create a VarData object for an OpenMDAO variable named `name`.

`VarData` objects are used to construct arguments to OpenMDAO's `Component.add_input` and `Component.add_output` methods.
Specifically, if a `VarData` object `var` refers to an input variable, the `Component.add_input` call will look like this:

```python
Component.add_input(var.name, shape=var.shape, val=var.val, units=var.units, tags=var.tags, shape_by_conn=var.shape_by_conn)
```

and if the `VarData` object `var` is an output variable, the `Component.add_output` call will look like this:

```python
Component.add_output(var.name, shape=var.shape, val=var.val, units=var.units, lower=var.lower, upper=var.upper, tags=var.tags, shape_by_conn=var.shape_by_conn)
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
- `shape_by_conn::Bool = false`: if `true`, shape this variable by its connected output (if an input) or input (if an output)
- `copy_shape::Union{String,Nothing} = nothing`: if a string, shape this variable by the local variable indicated by `copy_shape`
"""
struct VarData{N,TVal<:Union{Float64,<:AbstractArray{Float64,N}},TUnits<:Union{String,Nothing},TBounds<:Union{Float64,<:AbstractArray{Float64,N}, Nothing}, TTags<:Union{<:AbstractVector{String},Nothing},TCS<:Union{String,Nothing}}
    name::String
    val::TVal
    shape::NTuple{N,Int64}
    units::TUnits
    lower::TBounds
    upper::TBounds
    tags::TTags
    shape_by_conn::Bool
    copy_shape::TCS

    function VarData(name, val::TVal, shape::NTuple{N,Int64}, units, lower::TBounds, upper::TBounds, tags::TTags=nothing, shape_by_conn=false, copy_shape::TCS=nothing) where {N,TVal<:Union{Float64,<:AbstractArray{Float64,N}},TBounds,TTags,TCS}
        _vd_checkbounds(val, shape) || throw(ArgumentError("size of val argument $(size(val)) should match shape argument $(shape)"))
        _vd_checkbounds(lower, shape) || throw(ArgumentError("size of lower argument $(size(lower)) should match shape argument $(shape)"))
        _vd_checkbounds(upper, shape) || throw(ArgumentError("size of upper argument $(size(upper)) should match shape argument $(shape)"))
        return new{N,typeof(val),typeof(units),TBounds,TTags,TCS}(name, val, shape, units, lower, upper, tags, shape_by_conn, copy_shape)
    end
end

VarData(name; val=nothing, shape=nothing, units=nothing, lower=nothing, upper=nothing, tags=nothing, shape_by_conn=false, copy_shape=nothing) = VarData(name, val, shape, units, lower, upper, tags, shape_by_conn, copy_shape)

VarData(name, val::Float64, shape::Nothing, units, lower, upper, tags=nothing, shape_by_conn=false, copy_shape=nothing) = VarData(name, val, (1,), units, lower, upper, tags, shape_by_conn, copy_shape)
VarData(name, val::AbstractArray{Float64,N}, shape::Nothing, units, lower, upper, tags=nothing, shape_by_conn=false, copy_shape=nothing) where {N} = VarData(name, val, size(val), units, lower, upper, tags, shape_by_conn, copy_shape)
VarData(name, val::Nothing, shape::Nothing, units, lower, upper, tags=nothing, shape_by_conn=false, copy_shape=nothing) = VarData(name, 1.0, (1,), units, lower, upper, tags, shape_by_conn, copy_shape)

VarData(name, val::Float64, shape::Int64, units, lower, upper, tags=nothing, shape_by_conn=false, copy_shape=nothing) = VarData(name, val, (shape,), units, lower, upper, tags, shape_by_conn, copy_shape)
VarData(name, val::AbstractArray{Float64,N}, shape::Int64, units, lower, upper, tags=nothing, shape_by_conn=false, copy_shape=nothing) where {N} = VarData(name, val, (shape,), units, lower, upper, tags, shape_by_conn, copy_shape)
VarData(name, val::Nothing, shape::Int64, units, lower, upper, tags=nothing, shape_by_conn=false, copy_shape=nothing) = VarData(name, 1.0, (shape,), units, lower, upper, tags, shape_by_conn, copy_shape)

# These next two are commented out because they're handled by the `VarData` inner constructor.
# VarData(name, val::Float64, shape::NTuple{N,Int64}, units, lower, upper) where {N} = VarData(name, val, shape, units, lower, upper)
# VarData(name, val::AbstractArray{Float64,N}, shape::NTuple{N,Int64}, units, lower, upper) where {N} = VarData(name, val, shape, units, lower, upper)
VarData(name, val::Nothing, shape::NTuple{N,Int64}, units, lower, upper, tags=nothing, shape_by_conn=false, copy_shape=nothing) where {N} = VarData(name, 1.0, shape, units, lower, upper, tags, shape_by_conn, copy_shape)

