module OpenMDAOCore

using ComponentArrays: ComponentVector, ComponentMatrix, getaxes, getdata
using SparseArrays: sparse, findnz
using SparseDiffTools: forwarddiff_color_jacobian! 

include("utils.jl")

export VarData, PartialsData, AbstractComp, AbstractExplicitComp, AbstractImplicitComp
export AbstractAutoSparseForwardDiffExplicitComp
export has_setup_partials
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

function setup_partials(self::AbstractComp, input_sizes, output_sizes)
    throw(OpenMDAOMethodError(self, "setup_partials"))
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

function has_setup_partials(self::AbstractComp)
    # First, figure out which method would be called with self.
    T = typeof(self)
    specific = which(setup_partials, (T, Any, Any))
    # Next, get the fallback method.
    fallback = which(setup_partials, (AbstractComp, Any, Any))
    # If the specific method isn't the fallback method, then the user must have
    # implemented it, so return true.
    return !(specific === fallback)
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

abstract type AbstractAutoSparseForwardDiffExplicitComp <: AbstractExplicitComp end

get_callback(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.compute_forwarddiffable!
get_input_ca(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.X_ca
get_output_ca(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.Y_ca
get_sparse_jacobian_ca(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.J_ca_sparse
get_sparse_jacobian_cache(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.jac_cache
get_units(comp::AbstractAutoSparseForwardDiffExplicitComp, varname) = comp.units_dict[varname]

function get_input_var_data(self::AbstractAutoSparseForwardDiffExplicitComp)
    ca = get_input_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k)) for k in keys(ca)]
end

function get_output_var_data(self::AbstractAutoSparseForwardDiffExplicitComp)
    ca = get_output_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k)) for k in keys(ca)]
end

function get_partials_data(self::AbstractAutoSparseForwardDiffExplicitComp)
    rcdict = get_rows_cols_dict(get_sparse_jacobian_ca(self))
    partials_data = Vector{OpenMDAOCore.PartialsData}()
    for (output_name, input_name) in keys(rcdict)
        rows, cols = rcdict[output_name, input_name]
        # Convert from 1-based to 0-based indexing.
        rows0based = rows .- 1
        cols0based = cols .- 1
        push!(partials_data, OpenMDAOCore.PartialsData(string(output_name), string(input_name); rows=rows0based, cols=cols0based))
    end

    return partials_data
end

function OpenMDAOCore.setup(self::AbstractAutoSparseForwardDiffExplicitComp)
    input_data = get_input_var_data(self)
    output_data = get_output_var_data(self)
    partials_data = get_partials_data(self)

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::AbstractAutoSparseForwardDiffExplicitComp, inputs, outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Call the actual function.
    Y_ca = get_output_ca(self)
    f! = get_callback(self)
    f!(Y_ca, X_ca)

    # Copy the output `ComponentArray` to the outputs.
    for oname in keys(Y_ca)
        outputs[string(oname)] .= @view(Y_ca[oname])
    end

    return nothing
end

function OpenMDAOCore.compute_partials!(self::AbstractAutoSparseForwardDiffExplicitComp, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Get the Jacobian.
    f! = get_callback(self)
    J_ca_sparse = get_sparse_jacobian_ca(self)
    jac_cache = get_sparse_jacobian_cache(self)
    forwarddiff_color_jacobian!(J_ca_sparse, f!, X_ca, jac_cache)

    # Extract the derivatives from `J_ca_sparse` and put them in `partials`.
    raxis, caxis = getaxes(J_ca_sparse)
    for oname in keys(raxis)
        for iname in keys(caxis)
            partials[string(oname), string(iname)] .= @view(J_ca_sparse[oname, iname])
        end
    end

    return nothing
end

has_setup_partials(self::AbstractAutoSparseForwardDiffExplicitComp) = false
has_compute_partials(self::AbstractAutoSparseForwardDiffExplicitComp) = true
has_compute_jacvec_product(self::AbstractAutoSparseForwardDiffExplicitComp) = false

end # module
