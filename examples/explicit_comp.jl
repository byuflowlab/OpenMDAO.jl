using OpenMDAO
using PyCall

juila_comps = pyimport("omjl.julia_comps")
om = pyimport("openmdao.api")

function square_it_compute!(options, inputs, outputs)
    a = options["a"]
    x = inputs["x"]
    y = inputs["y"]
    @. outputs["z1"] = a*x*x + y*y
    @. outputs["z2"] = a*x + y
end

square_it_compute2! = pyfunction(square_it_compute!,
                                PyDict{String, PyAny},
                                PyDict{String, PyArray},
                                PyDict{String, PyArray})

function square_it_compute_partials!(options, inputs, partials)
    a = options["a"]
    x = inputs["x"]
    y = inputs["y"]
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.
end

square_it_compute_partials2! = pyfunction(square_it_compute_partials!,
                                          PyDict{String, PyAny},
                                          PyDict{String, PyArray},
                                          PyDict{Tuple{String, String}, PyArray})

options_data = [OM_Options_Data("a", Int, 2)]
input_data = [OM_Var_Data("x", 1, [2.0]), OM_Var_Data("y", 1, [3.0])]
output_data = [OM_Var_Data("z1", 1, [2.0]), OM_Var_Data("z2", 1, [3.0])]
partials_data = [
                 OM_Partials_Data("z1", "x")
                 OM_Partials_Data("z1", "y")
                 OM_Partials_Data("z2", "x")
                 OM_Partials_Data("z2", "y")
                ]

square_it_data = OM_Ecomp_Data(square_it_compute2!,
                               square_it_compute_partials2!,
                               options_data,
                               input_data,
                               output_data,
                               partials_data)

prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 2.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

comp = juila_comps.JuliaExplicitComp(julia_comp_data=square_it_data)
prob.model.add_subsystem("square_it_comp", comp, promotes=["*"])

prob.setup()
prob.final_setup()
prob.run_model()
@show prob.get_val("z1")
@show prob.get_val("z2")
@show prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
