abstract type AbstractComp end
abstract type AbstractExplicitComp <: AbstractComp end
abstract type AbstractImplicitComp <: AbstractComp end

"""
    OpenMDAOCoreError

Abstract supertype for all exceptions thrown by OpenMDAOCore.jl, so that user code can catch any of them with a single `catch OpenMDAOCoreError`.
"""
abstract type OpenMDAOCoreError <: Exception end

"""
    OpenMDAOMethodError{T}

Exception thrown by the fallback (i.e. not-implemented-by-the-user) OpenMDAO interface methods, like `setup` and `compute!`, when they are called on a component that does not implement them.
(These methods should only be called on components that implement them — OpenMDAO, via omjlcomps, uses the [`has_compute_partials`](@ref)-style functions to check — so getting this error usually means a component was called through an interface it doesn't support.)
"""
struct OpenMDAOMethodError{T} <: OpenMDAOCoreError
    method_name::String
end

function OpenMDAOMethodError(self::AbstractComp, method_name)
    T = typeof(self)
    return OpenMDAOMethodError{T}(method_name)
end

function Base.showerror(io::IO, e::OpenMDAOMethodError{T}) where {T}
    print(io, "called fallback $(e.method_name) method for type $T")
end

"""
    OpenMDAOPrepNotCreatedError

Exception thrown when a derivative-computing method (e.g. `compute_partials!`, `linearize!`, `compute_jacvec_product!`, or `apply_linear!`) is called on an AD component whose DifferentiationInterface.jl derivative preparation has not been created yet.
This happens when the component is created with `force_skip_prep=true`, or with `shape_by_conn`/`copy_shape` variables, before OpenMDAO has finished setting up the problem (the latter case is resolved automatically during problem setup).
"""
struct OpenMDAOPrepNotCreatedError <: OpenMDAOCoreError
    comp
    method_name::String
end

function Base.showerror(io::IO, e::OpenMDAOPrepNotCreatedError)
    print(io, """
        The derivative preparation for this component of type $(typeof(e.comp)) has not been created, so $(e.method_name) cannot be called. This happens when the component is created with `force_skip_prep=true` or with `shape_by_conn`/`copy_shape` variables before OpenMDAO has finished setting up the problem (the latter case is resolved automatically during problem setup).""")
end

# Convenience constructor for the derivative-method guards.
function _prep_not_created_error(comp, method_name)
    return OpenMDAOPrepNotCreatedError(comp, method_name)
end

"""
    setup(self::AbstractComp)

Required method for user-defined `<:AbstractComp`s, called by OpenMDAO during problem setup.
Its job is to declare the component's variables and partial derivatives.

# Arguments
* `self`: the component.

# Returns
A `Tuple` of three `Vector`s:
* `input_data`: a `Vector` of [`VarData`](@ref), one entry per input variable.
* `output_data`: a `Vector` of [`VarData`](@ref), one entry per output (or state) variable.
* `partials_data`: a `Vector` of [`PartialsData`](@ref) describing the sparsity (and, optionally, the values) of the component's sub-Jacobians. This `Vector` may be empty if the partials are declared later, in `setup_partials`.

!!! note
    The AD components (`ADExplicitComp`/`ADImplicitComp`) implement this method themselves, building the variable data from their input and output `ComponentVector`s.
"""
function setup(self::AbstractComp)
    throw(OpenMDAOMethodError(self, "setup"))
    return nothing
end

"""
    setup_partials(self::AbstractComp, input_sizes, output_sizes)

Optional method for user-defined `<:AbstractComp`s, called by OpenMDAO during problem setup, *after* the sizes of all variables are known.
Implement this method if the partial derivatives of the component depend on the sizes of its variables — e.g. because some variables were declared with `shape_by_conn` and/or `copy_shape` — or if the component's internals need updating based on those sizes.
(OpenMDAO only calls this method if the component implements it; see [`has_setup_partials`](@ref).)

# Arguments
* `self`: the component.
* `input_sizes`: `Dict` mapping each input variable name (as a `String`) to its shape, as an `NTuple{N, Int}`.
* `output_sizes`: `Dict` mapping each output variable name (as a `String`) to its shape, as an `NTuple{N, Int}`.

# Returns
A `Tuple` of:
* `self`: the component to use going forward, which may be a modified copy of the input `self` (e.g. if internal arrays need to be resized).
* `partials_data`: a `Vector` of [`PartialsData`](@ref) describing the sparsity (and, optionally, the values) of the component's sub-Jacobians.

!!! note
    The AD components (`ADExplicitComp`/`ADImplicitComp`) implement this method themselves: their `setup_partials` creates the DifferentiationInterface.jl derivative preparation at the resolved variable sizes, so users of those components don't need to provide one.
"""
function setup_partials(self::AbstractComp, input_sizes, output_sizes)
    throw(OpenMDAOMethodError(self, "setup_partials"))
    return nothing
end

"""
    compute!(self::AbstractExplicitComp, inputs, outputs)

Required method for user-defined `<:AbstractExplicitComp`s: computes the component's outputs from the given inputs, in place.

# Arguments
* `self`: the component.
* `inputs`: a `Dict`-like object mapping the input variable names (`String`s) to their current values.
* `outputs`: a `Dict`-like object mapping the output variable names (`String`s) to writable storage for the output values.

# Returns
`nothing`. (Any returned value is ignored.)

!!! note
    The AD components implement this method by calling their user-defined function.
"""
function compute!(self::AbstractExplicitComp, inputs, outputs)
    throw(OpenMDAOMethodError(self, "compute!"))
    return nothing
end

"""
    compute_partials!(self::AbstractExplicitComp, inputs, partials)

Optional method for user-defined `<:AbstractExplicitComp`s, called by OpenMDAO when it needs the partial derivatives of the outputs with respect to the inputs, e.g. when assembling Jacobians.
(OpenMDAO only calls this method if the component implements it; see [`has_compute_partials`](@ref).)

# Arguments
* `self`: the component.
* `inputs`: a `Dict`-like object mapping the input variable names (`String`s) to their current values.
* `partials`: a two-level, `Dict`-like object, indexed as `partials["of", "wrt"]`, whose entries are writable arrays holding the sub-Jacobian of output `"of"` with respect to input `"wrt"`. The shapes of these arrays are as declared in `setup` (or `setup_partials`).

# Returns
`nothing`.
"""
function compute_partials!(self::AbstractExplicitComp, inputs, partials)
    throw(OpenMDAOMethodError(self, "compute_partials!"))
    return nothing
end

"""
    compute_jacvec_product!(self::AbstractExplicitComp, inputs, d_inputs, d_outputs, mode)

Optional method for user-defined `<:AbstractExplicitComp`s that implement OpenMDAO's "matrix-free" API: instead of assembling sub-Jacobians in `compute_partials!`, the component computes Jacobian-vector products on the fly.
(OpenMDAO only calls this method if the component implements it; see [`has_compute_jacvec_product`](@ref).)

# Arguments
* `self`: the component.
* `inputs`: a `Dict`-like object mapping the input variable names (`String`s) to their current values.
* `d_inputs`: a `Dict`-like object mapping the input variable names to the incoming derivative ("seed") values.
* `d_outputs`: a `Dict`-like object mapping the output variable names to writable storage for the resulting derivative values.
* `mode`: either `"fwd"`, in which case the product of the Jacobian with `d_inputs` is added to `d_outputs`, or `"rev"`, in which case the product of the transposed Jacobian with `d_outputs` is added to `d_inputs`.

# Returns
`nothing`.
"""
function compute_jacvec_product!(self::AbstractExplicitComp, inputs, d_inputs, d_outputs, mode)
    throw(OpenMDAOMethodError(self, "compute_jacvec_product!"))
    return nothing
end

"""
    apply_nonlinear!(self::AbstractImplicitComp, inputs, outputs, residuals)

Required method for user-defined `<:AbstractImplicitComp`s: evaluates the residuals of the implicit equations that the component defines, given the current inputs and states.

# Arguments
* `self`: the component.
* `inputs`: a `Dict`-like object mapping the input variable names (`String`s) to their current values.
* `outputs`: a `Dict`-like object mapping the state (output) variable names (`String`s) to their current values.
* `residuals`: a `Dict`-like object mapping the state (output) variable names to writable storage for the residual values.

# Returns
`nothing`.

!!! note
    The AD components implement this method by calling their user-defined residual function.
"""
function apply_nonlinear!(self::AbstractImplicitComp, inputs, outputs, residuals)
    throw(OpenMDAOMethodError(self, "apply_nonlinear!"))
    return nothing
end

"""
    solve_nonlinear!(self::AbstractImplicitComp, inputs, outputs)

Optional method for user-defined `<:AbstractImplicitComp`s: solves the component's implicit equations for the states (`outputs`), given the `inputs`.
If the component does not implement this method, OpenMDAO's nonlinear solvers take care of solving the implicit system instead.
(OpenMDAO only calls this method if the component implements it; see [`has_solve_nonlinear`](@ref).)

# Arguments
* `self`: the component.
* `inputs`: a `Dict`-like object mapping the input variable names (`String`s) to their current values.
* `outputs`: a `Dict`-like object mapping the state (output) variable names (`String`s) to writable storage for the solved state values.

# Returns
`nothing`.
"""
function solve_nonlinear!(self::AbstractImplicitComp, inputs, outputs)
    throw(OpenMDAOMethodError(self, "solve_nonlinear!"))
    return nothing
end

"""
    linearize!(self::AbstractImplicitComp, inputs, outputs, partials)

Optional method for user-defined `<:AbstractImplicitComp`s, called by OpenMDAO when it needs the partial derivatives of the residuals with respect to the inputs *and* the states, e.g. when assembling Jacobians.
(OpenMDAO only calls this method if the component implements it; see [`has_linearize`](@ref).)

# Arguments
* `self`: the component.
* `inputs`: a `Dict`-like object mapping the input variable names (`String`s) to their current values.
* `outputs`: a `Dict`-like object mapping the state (output) variable names (`String`s) to their current values.
* `partials`: a two-level, `Dict`-like object, indexed as `partials["of", "wrt"]`, whose entries are writable arrays holding the sub-Jacobian of residual `"of"` with respect to input or state `"wrt"`.

# Returns
`nothing`.
"""
function linearize!(self::AbstractImplicitComp, inputs, outputs, partials)
    throw(OpenMDAOMethodError(self, "linearize!"))
    return nothing
end

"""
    apply_linear!(self::AbstractImplicitComp, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)

Optional method for user-defined `<:AbstractImplicitComp`s that implement OpenMDAO's "matrix-free" API: instead of assembling sub-Jacobians in `linearize!`, the component computes Jacobian-vector products on the fly.
(OpenMDAO only calls this method if the component implements it; see [`has_apply_linear`](@ref).)

# Arguments
* `self`: the component.
* `inputs`: a `Dict`-like object mapping the input variable names (`String`s) to their current values.
* `outputs`: a `Dict`-like object mapping the state (output) variable names (`String`s) to their current values.
* `d_inputs`: a `Dict`-like object mapping the input variable names to (writable) storage for the input derivative values.
* `d_outputs`: a `Dict`-like object mapping the state (output) variable names to (writable) storage for the state derivative values.
* `d_residuals`: a `Dict`-like object mapping the state (output) variable names to the residual derivative ("seed") values.
* `mode`: either `"fwd"`, in which case the product of the Jacobian with `d_inputs` and `d_outputs` is added to `d_residuals`, or `"rev"`, in which case the product of the transposed Jacobian with `d_residuals` is added to `d_inputs` and `d_outputs`.

# Returns
`nothing`.
"""
function apply_linear!(self::AbstractImplicitComp, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    throw(OpenMDAOMethodError(self, "apply_linear!"))
    return nothing
end

"""
    solve_linear!(self::AbstractImplicitComp, d_outputs, d_residuals, mode)

Optional method for user-defined `<:AbstractImplicitComp`s: solves the linearized system associated with the component's implicit equations.
For `mode="fwd"`, the method solves for `d_outputs`, given `d_residuals`, such that the residual derivatives vanish; for `mode="rev"`, it solves for `d_residuals`, given `d_outputs`.
If the component does not implement this method, OpenMDAO's linear solvers take care of the solve, typically using the Jacobian assembled from `linearize!`.
(OpenMDAO only calls this method if the component implements it; see [`has_solve_linear`](@ref).)

# Arguments
* `self`: the component.
* `d_outputs`: a `Dict`-like object mapping the state (output) variable names to the state derivative values (for `mode="rev"`) or writable storage for them (for `mode="fwd"`).
* `d_residuals`: a `Dict`-like object mapping the state (output) variable names to the residual derivative values (for `mode="fwd"`) or writable storage for them (for `mode="rev"`).
* `mode`: either `"fwd"` or `"rev"`.

# Returns
`nothing`.
"""
function solve_linear!(self::AbstractImplicitComp, d_outputs, d_residuals, mode)
    throw(OpenMDAOMethodError(self, "solve_linear!"))
    return nothing
end

"""
    guess_nonlinear!(self::AbstractImplicitComp, inputs, outputs, residuals)

Optional method for user-defined `<:AbstractImplicitComp`s, called by OpenMDAO before a nonlinear solve, giving the component a chance to provide initial guesses for the states (`outputs`), based on the `inputs`.
If the component does not implement this method, the values declared in `setup` are used as the initial guesses.
(OpenMDAO only calls this method if the component implements it; see [`has_guess_nonlinear`](@ref).)

Note that, although OpenMDAO computes the current residuals and passes them in, the residual vector is treated as read-only in `guess_nonlinear`, so the `residuals` argument is provided for context only; any changes to it are ignored.

# Arguments
* `self`: the component.
* `inputs`: a `Dict`-like object mapping the input variable names (`String`s) to their current values.
* `outputs`: a `Dict`-like object mapping the state (output) variable names (`String`s) to writable storage for the initial guesses.
* `residuals`: a `Dict`-like object mapping the state (output) variable names to their current, freshly-computed residuals. OpenMDAO treats the residuals as read-only in `guess_nonlinear`, so this argument is for context only.

# Returns
`nothing`.
"""
function guess_nonlinear!(self::AbstractImplicitComp, inputs, outputs, residuals)
    throw(OpenMDAOMethodError(self, "guess_nonlinear!"))
    return nothing
end

"""
    has_setup_partials(self::AbstractComp)

Return `true` if the component implements the setup_partials method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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

"""
    has_compute_partials(self::AbstractExplicitComp)

Return `true` if the component implements the `compute_partials!` method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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

"""
    has_compute_jacvec_product(self::AbstractExplicitComp)

Return `true` if the component implements the `compute_jacvec_product!` method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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

"""
    has_apply_nonlinear(self::AbstractImplicitComp)

Return `true` if the component implements the `apply_nonlinear!` method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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

"""
    has_solve_nonlinear(self::AbstractImplicitComp)

Return `true` if the component implements the `solve_nonlinear!` method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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

"""
    has_linearize(self::AbstractImplicitComp)

Return `true` if the component implements the `linearize!` method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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

"""
    has_apply_linear(self::AbstractImplicitComp)

Return `true` if the component implements the `apply_linear!` method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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

"""
    has_solve_linear(self::AbstractImplicitComp)

Return `true` if the component implements the `solve_linear!` method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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

"""
    has_guess_nonlinear(self::AbstractImplicitComp)

Return `true` if the component implements the `guess_nonlinear!` method — i.e. if the method that would be called on `self` differs from the fallback defined in OpenMDAOCore — and `false` otherwise.
omjlcomps uses these functions to determine which OpenMDAO methods need to be forwarded to Julia.
"""
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
