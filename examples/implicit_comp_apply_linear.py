import openmdao.api as om
import julia.Main as julia
from julia.OpenMDAO import make_component


julia.include("components/implicit_with_apply_linear.jl")

prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("a", 1.0)
ivc.add_output("b", -4.0)
ivc.add_output("c", 3.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(julia.ImplicitWithApplyLinear())
comp.linear_solver = om.DirectSolver(assemble_jac=False)
comp.nonlinear_solver = om.NewtonSolver(
    solve_subsystems=True, iprint=2, err_on_non_converge=True)
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
print(prob.get_val("x"))
print(prob.compute_totals(of=["x"], wrt=["a", "b", "c"]))
