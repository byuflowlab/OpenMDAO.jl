using OpenMDAO
using PyCall

juila_comps = pyimport("omjl.julia_comps")
om = pyimport("openmdao.api")

function compute_square_it(inputs, outputs)
    x = inputs["x"]
    y = inputs["y"]
    @. outputs["z1"] = x*x + y*y
    @. outputs["z2"] = x + y
end

compute_square_it2 = pyfunction(compute_square_it,
                                PyDict{String, PyArray},
                                PyDict{String, PyArray})

function compute_partials_square_it(inputs, partials)
    x = inputs["x"]
    y = inputs["y"]
    @. partials["z1", "x"] = 2*x
    @. partials["z1", "y"] = 2*y
    @. partials["z2", "x"] = 1.
    @. partials["z2", "y"] = 1.
end

compute_partials_square_it2 = pyfunction(compute_partials_square_it,
                                         PyDict{String, PyArray},
                                         PyDict{Tuple{String, String}, PyArray})

input_data = [OM_Var_Data("x", 1, [2.0]), OM_Var_Data("y", 1, [3.0])]
output_data = [OM_Var_Data("z1", 1, [2.0]), OM_Var_Data("z2", 1, [3.0])]
partials_data = [
                 OM_Partials_Data("z1", "x")
                 OM_Partials_Data("z1", "y")
                 OM_Partials_Data("z2", "x")
                 OM_Partials_Data("z2", "y")
                ]

square_it_data = OM_Ecomp_Data(compute_square_it2,
                               compute_partials_square_it2,
                               input_data,
                               output_data,
                               partials_data)

explicit_comp = juila_comps.JuliaExplicitComp(julia_comp_data=square_it_data)

prob = om.Problem()

ivc = om.IndepVarComp()
ivc.add_output("x", 2.0)
ivc.add_output("y", 2.0)
prob.model.add_subsystem("ivc", ivc, promotes=["*"])

prob.model.add_subsystem("square_it_comp", explicit_comp, promotes=["*"])

prob.setup()
prob.final_setup()
explicit_comp.setup()
prob.run_model()
@show prob.get_val("z1")
@show prob.get_val("z2")
@show prob.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
