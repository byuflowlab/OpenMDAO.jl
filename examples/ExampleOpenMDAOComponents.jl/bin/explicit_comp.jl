using PyCall  # Apparently needed, despite not being used directly in this script.
using OpenMDAO: om, make_component
using ExampleOpenMDAOComponents: SimpleExplicit


prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 3.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = make_component(SimpleExplicit(4.0))  # Need to convert Julia obj to Python obj
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.run_model()
println(prob.get_val("z1"))
println(prob.get_val("z2"))
println(prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"]))
