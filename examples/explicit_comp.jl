using OpenMDAO
import PyCall

om = PyCall.pyimport("openmdao.api")

struct SquareIt{TF} <: AbstractExplicitComp
    a::TF
end


function OpenMDAO.setup(self::SquareIt)
    inputs = [
        VarData("x", [1], [2.0]),
        VarData("y", [1], [3.0])
    ]

    outputs = [
        VarData("z1", [1], [2.0]),
        VarData("z2", [1], [3.0])
    ]

    partials = [
        PartialsData("z1", "x"),
        PartialsData("z1", "y"),
        PartialsData("z2", "x"),
        PartialsData("z2", "y")
    ]

    return inputs, outputs, partials
end

function OpenMDAO.compute!(self::SquareIt, inputs, outputs)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = a*x*x + y*y
    @. outputs["z2"] = a*x + y
end

function OpenMDAO.compute_partials!(self::SquareIt, inputs, partials)
    a = self.a
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

comp = make_component(SquareIt(4.0))
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.run_model()
@show prob.get_val("z1")
@show prob.get_val("z2")
@show prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
