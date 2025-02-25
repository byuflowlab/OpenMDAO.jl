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

get_aviary_input_name(comp::AbstractComp, ca_name::Symbol) = string(ca_name)
get_aviary_input_name(comp::AbstractComp, ca_name::AbstractString) = ca_name

get_aviary_output_name(comp::AbstractComp, ca_name::Symbol) = string(ca_name)
get_aviary_output_name(comp::AbstractComp, ca_name::AbstractString) = ca_name
