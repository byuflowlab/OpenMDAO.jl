using OpenMDAO: om, make_component

include("components/implicit_with_apply_linear.jl")

prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("a", 1.0)
ivc.add_output("b", -4.0)
ivc.add_output("c", 3.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(ImplicitWithApplyLinear())
comp.linear_solver = om.DirectSolver(assemble_jac=false)
comp.nonlinear_solver = om.NewtonSolver(
    solve_subsystems=true, iprint=2, err_on_non_converge=true)
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
println(prob.get_val("x"))
println(prob.compute_totals(of=["x"], wrt=["a", "b", "c"]))
