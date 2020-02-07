using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, om, make_component, detect_compute_partials
import OpenMDAO: detect_compute_partials

struct SimpleExplicit{TF} <: AbstractExplicitComp
    a::TF  # these would be like "options" in openmdao
end


function OpenMDAO.setup(::SimpleExplicit)
    # Trying out different combinations of tuple vs scalar for shape, and array
    # vs scalar for val.
    inputs = [
        VarData("x", shape=1, val=[2.0]),
        VarData("y", shape=1, val=3.0)
    ]

    outputs = [
        VarData("z1", shape=(1,), val=[2.0]),
        VarData("z2")  #default is just 1.0  , shape=(1,), val=3.0)
    ]

    partials = [
        PartialsData("z1", "x"),
        PartialsData("z1", "y"),
        PartialsData("z2", "x"),
        PartialsData("z2", "y")
    ]

    return inputs, outputs, partials
end

function OpenMDAO.compute!(square::SimpleExplicit, inputs, outputs)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = a*x*x + y*y  # change arrays in-place, thus dot syntax
    @. outputs["z2"] = a*x + y
end

prob = om.Problem()

ec = SimpleExplicit(4.0)
comp = make_component(ec)  # Need to convert Julia obj to Python obj
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

# So you use @test_warn to test if a function uses @warn to issue a warning,
# right? WRONG!
# https://github.com/JuliaLang/julia/issues/28549
# https://github.com/JuliaLang/julia/issues/25612
# msg = "Warning: No compute_partials! method found for $(typeof(ec))"
# @test_warn msg prob.setup()

# Check that a warning is issued.
msg = "No compute_partials! method found for $(typeof(ec))" 
@test_logs (:warn, msg) prob.setup()

# Check that no warning is issued if we tell OpenMDAO.jl to not look for
# compute_partials!.
detect_compute_partials(::Type{<:SimpleExplicit}) = false
@test_logs prob.setup()

function OpenMDAO.compute_partials!(square::SimpleExplicit, inputs, partials)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0
end

# Giving an empty (:stream, msg) tuple to @test_logs is how you test that no
# logging message is issued.
detect_compute_partials(::Type{<:SimpleExplicit}) = true
@test_logs prob.setup()

# Test that we don't get a warning when the compute_partials! is defined, but we
# tell OpenMDAO.jl to not look for it anyway.
detect_compute_partials(::Type{<:SimpleExplicit}) = false
@test_logs prob.setup()
