using OpenMDAO
using PyCall

juila_comps = pyimport("omjl.julia_comps")
om = pyimport("openmdao.api")

function square_it_apply_nonlinear!(options, inputs, outputs, residuals)
    a = options["a"]
    x = inputs["x"]
    y = inputs["y"]
    @. residuals["z1"] = outputs["z1"] - (a*x*x + y*y)
    @. residuals["z2"] = outputs["z2"] - (a*x + y)
end

square_it_apply_nonlinear2! = pyfunction(square_it_apply_nonlinear!,
                                         PyDict{String, PyAny},
                                         PyDict{String, PyArray},
                                         PyDict{String, PyArray},
                                         PyDict{String, PyArray})

function square_it_linearize!(options, inputs, outputs, partials)
    a = options["a"]
    x = inputs["x"]
    y = inputs["y"]
    @. partials["z1", "z1"] = 1.0
    @. partials["z1", "x"] = -2*a*x
    @. partials["z1", "y"] = -2*y

    @. partials["z2", "z2"] = 1.0
    @. partials["z2", "x"] = -a
    @. partials["z2", "y"] = -1.0
end

square_it_linearize2! = pyfunction(square_it_linearize!,
                                   PyDict{String, PyAny},
                                   PyDict{String, PyArray},
                                   PyDict{String, PyArray},
                                   PyDict{Tuple{String, String}, PyArray})

options_data = [OM_Options_Data("a", Int, 2)]
input_data = [OM_Var_Data("x", 1, [2.0]), OM_Var_Data("y", 1, [3.0])]
output_data = [OM_Var_Data("z1", 1, [2.0]), OM_Var_Data("z2", 1, [3.0])]
partials_data = [
                 OM_Partials_Data("z1", "z1")
                 OM_Partials_Data("z1", "x")
                 OM_Partials_Data("z1", "y")
                 OM_Partials_Data("z2", "z2")
                 OM_Partials_Data("z2", "x")
                 OM_Partials_Data("z2", "y")
                ]

square_it_data = OM_Icomp_Data(square_it_apply_nonlinear2!,
                               square_it_linearize2!,
                               options_data,
                               input_data,
                               output_data,
                               partials_data)

prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 2.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = juila_comps.JuliaImplicitComp(julia_comp_data=square_it_data)
comp.linear_solver = om.DirectSolver(assemble_jac=true)
comp.nonlinear_solver = om.NewtonSolver()
comp.nonlinear_solver.options["solve_subsystems"] = true
comp.nonlinear_solver.options["iprint"] = 2
comp.nonlinear_solver.options["err_on_non_converge"] = true
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
@show prob.get_val("z1")
@show prob.get_val("z2")
@show prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
