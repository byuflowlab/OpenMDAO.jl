using PyCall  # Apparently needed, despite not being used directly in this script.
using OpenMDAO: om, make_component, SimpleImplicit


prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 2.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(SimpleImplicit(2.0))
comp.linear_solver = om.DirectSolver(assemble_jac=true)
comp.nonlinear_solver = om.NewtonSolver(
    solve_subsystems=true, iprint=2, err_on_non_converge=true)
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
println(prob.get_val("z1"))
println(prob.get_val("z2"))
println(prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"]))
