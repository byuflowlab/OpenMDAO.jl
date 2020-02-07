using OpenMDAO: AbstractImplicitComp, VarData, PartialsData, om, make_component
import OpenMDAO: detect_apply_nonlinear, detect_linearize, detect_guess_nonlinear, detect_solve_nonlinear, detect_apply_linear

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

prob = om.Problem()

ic = SimpleImplicit(4.0)
comp = make_component(ic)  # Need to convert Julia obj to Python obj
prob.model.add_subsystem("square_it_comp", comp)

detect_apply_nonlinear(::Type{<:SimpleImplicit}) = false
detect_linearize(::Type{<:SimpleImplicit}) = false
detect_guess_nonlinear(::Type{<:SimpleImplicit}) = false
detect_solve_nonlinear(::Type{<:SimpleImplicit}) = false
detect_apply_linear(::Type{<:SimpleImplicit}) = false

# Check that the warning is issued when the method doesn't exist.
detect_apply_nonlinear(::Type{<:SimpleImplicit}) = true
msg = "No apply_nonlinear! method found for $(typeof(ic))" 
@test_logs (:warn, msg) prob.setup()

# Check that the warning isn't issued when we tell OpenMDAO.jl not to look for it.
detect_apply_nonlinear(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

function OpenMDAO.apply_nonlinear!(square::SimpleImplicit, inputs, outputs, residuals)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = outputs["z1"] - (a*x*x + y*y)
    @. residuals["z2"] = outputs["z2"] - (a*x + y)
end

# Check that the warning isn't issued when the method exists and OpenMDAO.jl
# isn't looking for it.
detect_apply_nonlinear(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

# Check that the warning isn't issued when the method exists and OpenMDAO.jl is
# looking for it.
detect_apply_nonlinear(::Type{<:SimpleImplicit}) = true
@test_logs prob.setup()

# Check that the warning is issued when the method doesn't exist.
detect_linearize(::Type{<:SimpleImplicit}) = true
msg = "No linearize! method found for $(typeof(ic))" 
@test_logs (:warn, msg) prob.setup()

# Check that the warning isn't issued when we tell OpenMDAO.jl not to look for it.
detect_linearize(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

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

# Check that the warning isn't issued when the method exists and OpenMDAO.jl
# isn't looking for it.
detect_linearize(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

# Check that the warning isn't issued when the method exists and OpenMDAO.jl is
# looking for it.
detect_linearize(::Type{<:SimpleImplicit}) = true
@test_logs prob.setup()

# Check that the warning is issued when the method doesn't exist.
detect_guess_nonlinear(::Type{<:SimpleImplicit}) = true
msg = "No guess_nonlinear! method found for $(typeof(ic))" 
@test_logs (:warn, msg) prob.setup()

# Check that the warning isn't issued when we tell OpenMDAO.jl not to look for it.
detect_guess_nonlinear(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

function OpenMDAO.guess_nonlinear!(square::SimpleImplicit, inputs, outputs, residuals)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = 3.0
    @. outputs["z2"] = 4.0
end

# Check that the warning isn't issued when the method exists and OpenMDAO.jl
# isn't looking for it.
detect_guess_nonlinear(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

# Check that the warning isn't issued when the method exists and OpenMDAO.jl is
# looking for it.
detect_guess_nonlinear(::Type{<:SimpleImplicit}) = true
@test_logs prob.setup()

# Check that the warning is issued when the method doesn't exist.
detect_solve_nonlinear(::Type{<:SimpleImplicit}) = true
msg = "No solve_nonlinear! method found for $(typeof(ic))" 
@test_logs (:warn, msg) prob.setup()

# Check that the warning isn't issued when we tell OpenMDAO.jl not to look for it.
detect_solve_nonlinear(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

function OpenMDAO.solve_nonlinear!(square::SimpleImplicit, inputs, outputs)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = (a*x*x + y*y)
    @. outputs["z2"] = (a*x + y)
end

# Check that the warning isn't issued when the method exists and OpenMDAO.jl
# isn't looking for it.
detect_solve_nonlinear(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

# Check that the warning isn't issued when the method exists and OpenMDAO.jl is
# looking for it.
detect_solve_nonlinear(::Type{<:SimpleImplicit}) = true
@test_logs prob.setup()

# Check that the warning is issued when the method doesn't exist.
detect_apply_linear(::Type{<:SimpleImplicit}) = true
msg = "No apply_linear! method found for $(typeof(ic))" 
@test_logs (:warn, msg) prob.setup()

# Check that the warning isn't issued when we tell OpenMDAO.jl not to look for it.
detect_apply_linear(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

function OpenMDAO.apply_linear!(square::SimpleImplicit, inputs, outputs, d_inputs, d_ouputs, d_residuals, mode)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = (a*x*x + y*y)
    @. outputs["z2"] = (a*x + y)
end

# Check that the warning isn't issued when the method exists and OpenMDAO.jl
# isn't looking for it.
detect_apply_linear(::Type{<:SimpleImplicit}) = false
@test_logs prob.setup()

# Check that the warning isn't issued when the method exists and OpenMDAO.jl is
# looking for it.
detect_apply_linear(::Type{<:SimpleImplicit}) = true
@test_logs prob.setup()
