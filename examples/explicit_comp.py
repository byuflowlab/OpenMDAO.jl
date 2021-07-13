import openmdao.api as om
import julia.Main as julia
from julia.OpenMDAO import make_component


julia.include("components/simple_explicit.jl")

prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 3.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(julia.SimpleExplicit(4.0))
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.run_model()
print(prob.get_val("z1"))
print(prob.get_val("z2"))
print(prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"]))
