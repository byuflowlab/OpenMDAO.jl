using OpenMDAO: AbstractImplicitComp, VarData, PartialsData, om, make_component
import OpenMDAO: detect_apply_nonlinear, detect_linearize, detect_guess_nonlinear, detect_solve_nonlinear, detect_apply_linear

struct SimpleImplicit{TF} <: AbstractImplicitComp
    a::TF  # these would be like "options" in openmdao
end

detect_apply_nonlinear(::Type{<:SimpleImplicit}) = false
detect_linearize(::Type{<:SimpleImplicit}) = false
detect_guess_nonlinear(::Type{<:SimpleImplicit}) = false
detect_solve_nonlinear(::Type{<:SimpleImplicit}) = false
detect_apply_linear(::Type{<:SimpleImplicit}) = false

prob = om.Problem()

ic = SimpleImplicit(4.0)
comp = make_component(ic)  # Need to convert Julia obj to Python obj
prob.model.add_subsystem("square_it_comp", comp)

@test_throws PyJlError prob.setup()

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

# This shouldn't throw an error now.
prob.setup()
