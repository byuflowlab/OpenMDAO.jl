using OpenMDAO: AbstractExplicitComp, VarData, PartialsData, om, make_component, detect_compute_partials
import OpenMDAO: detect_compute_partials
using PyCall: PyError

struct SimpleExplicit{TF} <: AbstractExplicitComp
    a::TF  # these would be like "options" in openmdao
end

detect_compute_partials(::Type{<:SimpleExplicit}) = false

prob = om.Problem()

ec = SimpleExplicit(4.0)
comp = make_component(ec)  # Need to convert Julia obj to Python obj
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

@test_throws PyError prob.setup()

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

prob = om.Problem()

ec = SimpleExplicit(5.0)
comp = make_component(ec)  # Need to convert Julia obj to Python obj
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

@test_throws PyError prob.setup()

function OpenMDAO.compute!(square::SimpleExplicit, inputs, outputs)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = a*x*x + y*y  # change arrays in-place, thus dot syntax
    @. outputs["z2"] = a*x + y
end

prob = om.Problem()

ec = SimpleExplicit(6.0)
comp = make_component(ec)  # Need to convert Julia obj to Python obj
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

# Now this shouldn't throw an error.
prob.setup()
