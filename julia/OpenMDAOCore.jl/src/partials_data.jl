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

