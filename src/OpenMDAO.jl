module OpenMDAO
using PyCall

export ECompData, ICompData, OptionsData, VarData, PartialsData

function compute!(self::UnionAll, inputs, outputs)
    error("called dummy base compute! with self{$(typeof(self))}")
end

function compute_partials!(self::UnionAll, inputs, partials)
    error("called dummy base compute_partials! with self{$(typeof(self))}")
end

function apply_nonlinear!(self::UnionAll, inputs, outputs, residuals)
    error("called dummy base compute_partials! with self{$(typeof(self))}")
end

function linearize!(self::UnionAll, inputs, outputs, partials)
    error("called dummy base linearize! with self{$(typeof(self))}")
end

function guess_nonlinear!(self::UnionAll, inputs, outputs, residuals)
    error("called dummy base guess_nonlinear! with self{$(typeof(self))}")
end

struct ECompData
    self
    inputs
    outputs
    partials
    compute
    compute_partials
end

function ECompData(self, inputs, outputs; partials=nothing)

    # Create the Python wrapper of the compute! method.
    args = (typeof(self),  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray})  # outputs
    pycompute = pyfunction(compute!, args...)

    # Initialize pycompute_partials to a dummy value.
    local pycompute_partials = nothing

    # Check for the existance of a compute_partials! method for the `self` type.
    args = (typeof(self),  # self
            PyDict{String, PyArray},  # inputs
            PyDict{Tuple{String, String}, PyArray})  # partials
    try
        # Create the Python wrapper of the compute_partials! function.
        method = which(compute_partials!, args)
        pycompute_partials = pyfunction(compute_partials!, args...)
    catch err
        println("No compute_partials! method found for $(typeof(self))")
    end

    data = ECompData(self, inputs, outputs, partials, pycompute, pycompute_partials)
    return data
end

struct ICompData
    self
    inputs
    outputs
    partials
    apply_nonlinear
    linearize
    guess_nonlinear
end

function ICompData(self, inputs, outputs; partials=nothing)
    # Create the Python wrapper of the apply_nonlinear! method.
    args = (typeof(self),  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray},  # outputs
            PyDict{String, PyArray})  # residuals
    pyapply_nonlinear = pyfunction(apply_nonlinear!, args...)

    # Initialize pylinearize to a dummy value.
    local pylinearize = nothing
    args = (typeof(self),  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray},  # outputs
            PyDict{Tuple{String, String}, PyArray})  # partials

    try
        # Create the Python wrapper of the compute_partials! function.
        method = which(linearize!, args)
        pylinearize = pyfunction(linearize!, args...)
    catch err
        println("No linearize! method found for $(typeof(self))")
    end

    # Initialize pyguess_nonlinear to a dummy value.
    local pyguess_nonlinear = nothing
    args = (typeof(self),  # self
            PyDict{String, PyArray},  # inputs
            PyDict{String, PyArray},  # outputs
            PyDict{String, PyArray})  # residuals
    try
        # Create the Python wrapper of the compute_partials! function.
        method = which(guess_nonlinear!, args)
        pyguess_nonlinear = pyfunction(guess_nonlinear!, args...)
    catch err
        println("No guess_nonlinear! method found for $(typeof(self))")
    end

    data = ICompData(self, inputs, outputs, partials, pyapply_nonlinear, pylinearize, pyguess_nonlinear)
    return data
end

struct OptionsData
    name
    type
    val
end

struct VarData
    name
    shape
    val
    units
end

VarData(name, shape, val; units=nothing) = VarData(name, shape, val, units)

struct PartialsData
    of
    wrt
    rows
    cols
    val
end

PartialsData(of, wrt; rows=nothing, cols=nothing, val=nothing) = PartialsData(of, wrt, rows, cols, val)

end # module
