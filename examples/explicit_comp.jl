using OpenMDAO
using PyCall
import Base.convert

julia_comps = pyimport("omjl.julia_comps")
om = pyimport("openmdao.api")

struct SquareIt
    a
    inputs
    outputs
    partials
end

function SquareIt(a)
    inputs = [
        VarData("x", [1], [2.0]), 
        VarData("y", [1], [3.0])]
    outputs = [
        VarData("z1", [1], [2.0]), 
        VarData("z2", [1], [3.0])]
    partials = [
        PartialsData("z1", "x"),
        PartialsData("z1", "y"),
        PartialsData("z2", "x"),
        PartialsData("z2", "y")]
    return SquareIt(a, inputs, outputs, partials)
end

SquareIt() = SquareIt(2)

convert(::Type{SquareIt}, po::PyObject) = SquareIt(
    po.a, po.inputs, po.outputs, po.partials)

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
    @. partials["z2", "y"] = 1.
end


prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 2.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

# square_it_data = ECompData(SquareIt())
# comp = julia_comps.JuliaExplicitComp(julia_comp_data=square_it_data)
comp = julia_comps.JuliaExplicitComp(julia_comp_data=SquareIt())
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
@show prob.get_val("z1")
@show prob.get_val("z2")
@show prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
