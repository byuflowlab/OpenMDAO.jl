using OpenMDAO

struct SimpleImplicit{TF} <: AbstractImplicitComp
    a::TF  # these would be like "options" in openmdao
end

function OpenMDAO.setup(::SimpleImplicit)
 
    inputs = VarData[
        VarData("x", shape=(1,), val=[2.0]),
        VarData("y", shape=(1,), val=3.0)
    ]

    outputs = VarData[
        VarData("z1", shape=1, val=[2.0]),
        VarData("z2", shape=1, val=3.0)
    ]

    partials = PartialsData[
        PartialsData("z1", "x"),            
        PartialsData("z1", "y"),            
        PartialsData("z1", "z1"),           
        PartialsData("z2", "x"),            
        PartialsData("z2", "y"),            
        PartialsData("z2", "z2")
    ]

    return inputs, outputs, partials
end

# This tells OpenMDAO.jl that it shouldn't bother looking for a guess_nonlinear
# function for the SimpleImplicit type, because we didn't write one. If this
# line was removed (or the right-hand side was true), then it would print out a
# warning about not finding the guess_nonlinear function.
OpenMDAO.detect_guess_nonlinear(::Type{<:SimpleImplicit}) = false
OpenMDAO.detect_solve_nonlinear(::Type{<:SimpleImplicit}) = false
OpenMDAO.detect_apply_linear(::Type{<:SimpleImplicit}) = false

function OpenMDAO.apply_nonlinear!(square::SimpleImplicit, inputs, outputs, residuals)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = outputs["z1"] - (a*x*x + y*y)
    @. residuals["z2"] = outputs["z2"] - (a*x + y)
end

function OpenMDAO.linearize!(square::SimpleImplicit, inputs, outputs, partials)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = 1.0
    @. partials["z1", "x"] = -2*a*x
    @. partials["z1", "y"] = -2*y

    @. partials["z2", "z2"] = 1.0
    @. partials["z2", "x"] = -a
    @. partials["z2", "y"] = -1.0
end
