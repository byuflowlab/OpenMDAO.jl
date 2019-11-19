using OpenMDAO
import PyCall

struct SquareIt{TF} <: AbstractExplicitComp
    a::TF  # these would be like "options" in openmdao
end


function OpenMDAO.setup(::SquareIt)
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

function OpenMDAO.compute!(square::SquareIt, inputs, outputs)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = a*x*x + y*y  # change arrays in-place, thus dot syntax
    @. outputs["z2"] = a*x + y
end

function OpenMDAO.compute_partials!(square::SquareIt, inputs, partials)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0
end


prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 3.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(SquareIt(4.0))  # Need to convert Julia obj to Python obj
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.run_model()
println(prob.get_val("z1"))
println(prob.get_val("z2"))
println(prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"]))
