import openmdao.api as om
from omjl.julia_comps import JuliaExplicitComp
from julia.OpenMDAO import SimpleExplicit


prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 3.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = JuliaExplicitComp(julia_comp_data=SimpleExplicit(4.0))
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.run_model()
print(prob.get_val("z1"))
print(prob.get_val("z2"))
print(prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"]))
