using OpenMDAO

struct ImplicitWithApplyLinear <: AbstractImplicitComp
end

function OpenMDAO.setup(::ImplicitWithApplyLinear)
 
    inputs = [
        VarData("a", shape=(1,), val=[1.0]),
        VarData("b", shape=(1,), val=-4.0),
        VarData("c", val=3.0),
    ]

    outputs = [
        VarData("x", shape=(1,), val=5.0)
    ]

    partials = PartialsData[
        PartialsData("x", "a"),
        PartialsData("x", "b"),
        PartialsData("x", "c"),
    ]

    return inputs, outputs, partials
end

OpenMDAO.detect_linearize(::Type{<:ImplicitWithApplyLinear}) = false
OpenMDAO.detect_guess_nonlinear(::Type{<:ImplicitWithApplyLinear}) = false
OpenMDAO.detect_solve_nonlinear(::Type{<:ImplicitWithApplyLinear}) = false

function OpenMDAO.apply_nonlinear!(comp::ImplicitWithApplyLinear, inputs, outputs, residuals)
    a = inputs["a"]
    b = inputs["b"]
    c = inputs["c"]
    x = outputs["x"]
    @. residuals["x"] = a*x^2 + b*x + c
end

function OpenMDAO.apply_linear!(comp::ImplicitWithApplyLinear, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    a = inputs["a"]
    b = inputs["b"]
    c = inputs["c"]
    x = outputs["x"]
    if mode == "fwd"
        if haskey(d_residuals, "x")
            if haskey(d_outputs, "x")
                @. d_residuals["x"] += (2 * a * x + b) * d_outputs["x"]
            end
            if haskey(d_inputs, "a")
                @. d_residuals["x"] += x ^ 2 * d_inputs["a"]
            end
            if haskey(d_inputs, "b")
                @. d_residuals["x"] += x * d_inputs["b"]
            end
            if haskey(d_inputs, "c")
                @. d_residuals["x"] += d_inputs["c"]
            end
        end
    elseif mode == "rev"
        if haskey(d_residuals, "x")
            if haskey(d_outputs, "x")
                d_outputs["x"] += (2 * a * x + b) * d_residuals["x"]
            end
            if haskey(d_inputs, "a")
                d_inputs["a"] += x ^ 2 * d_residuals["x"]
            end
            if haskey(d_inputs, "b")
                d_inputs["b"] += x * d_residuals["x"]
            end
            if haskey(d_inputs, "c")
                d_inputs["c"] += d_residuals["x"]
            end
        end
    end
end
