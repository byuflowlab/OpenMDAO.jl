module OpenMDAO

export ECompData, ICompData, OptionsData, VarData, PartialsData

struct ECompData
    inputs
    outputs
    options
    partials
    compute
    compute_partials
end

ECompData(inputs, outputs; options=nothing, partials=nothing, compute=nothing, compute_partials=nothing) = ICompData(inputs, outputs, options, partials, compute, compute_partials)

struct ICompData
    inputs
    outputs
    options
    partials
    apply_nonlinear
    linearize
    guess_nonlinear
end

ICompData(inputs, outputs; options=nothing, partials=nothing, apply_nonlinear=nothing, linearize=nothing, guess_nonlinear=nothing) = ICompData(inputs, outputs, options, partials, apply_nonlinear, linearize, guess_nonlinear)

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
