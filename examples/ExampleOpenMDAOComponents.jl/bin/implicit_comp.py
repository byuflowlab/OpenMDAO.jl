import openmdao.api as om
from omjl.julia_comps import JuliaImplicitComp
from julia.ExampleOpenMDAOComponents import SimpleImplicit


prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 2.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = JuliaImplicitComp(julia_comp_data=SimpleImplicit(2.0))
comp.linear_solver = om.DirectSolver(assemble_jac=True)
comp.nonlinear_solver = om.NewtonSolver(
    solve_subsystems=True, iprint=2, err_on_non_converge=True)
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
print(prob.get_val("z1"))
print(prob.get_val("z2"))
print(prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"]))
