using OpenMDAO
using PyCall

julia_comps = pyimport("omjl.julia_comps")
om = pyimport("openmdao.api")

function actuator_disc_compute!(options, inputs, outputs)
        a = inputs["a"]
        Vu = inputs["Vu"]

        qA = @. .5 * inputs["rho"] * inputs["Area"] * Vu ^ 2

        @. outputs["Vd"] = Vu * (1 - 2 * a)
        Vd = outputs["Vd"]
        @. outputs["Vr"] = .5 * (Vu + Vd)

        @. outputs["Ct"] = 4 * a * (1 - a)
        Ct = outputs["Ct"]
        @. outputs["thrust"] = Ct * qA

        @. outputs["Cp"] = Ct * (1 - a)
        Cp = outputs["Cp"]
        @. outputs["power"] = Cp * qA * Vu
end

actuator_disc_compute2! = pyfunction(actuator_disc_compute!,
                                     PyDict{String, PyAny},
                                     PyDict{String, PyArray},
                                     PyDict{String, PyArray})

function actuator_disc_compute_partials!(options, inputs, J)
        a = inputs["a"]
        Vu = inputs["Vu"]
        Area = inputs["Area"]
        rho = inputs["rho"]

        # pre-compute commonly needed quantities
        a_times_area = @. a * Area
        one_minus_a = @. 1.0 - a
        a_area_rho_vu = @. a_times_area * rho * Vu

        @. J["Vr", "a"] = -Vu
        @. J["Vr", "Vu"] = one_minus_a

        @. J["Vd", "a"] = -2.0 * Vu

        @. J["Ct", "a"] = 4.0 - 8.0 * a

        @. J["thrust", "a"] = .5 * rho * Vu^2 * Area * J["Ct", "a"]
        @. J["thrust", "Area"] = 2.0 * Vu^2 * a * rho * one_minus_a
        @. J["thrust", "rho"] = 2.0 * a_times_area * Vu ^ 2 * (one_minus_a)
        @. J["thrust", "Vu"] = 4.0 * a_area_rho_vu * (one_minus_a)

        @. J["Cp", "a"] = 4.0 * a * (2.0 * a - 2.0) + 4.0 * (one_minus_a)^2

        @. J["power", "a"] = 2.0 * Area * Vu^3 * a * rho * ( 2.0 * a - 2.0) + 2.0 * Area * Vu^3 * rho * one_minus_a ^ 2
        @. J["power", "Area"] = 2.0 * Vu^3 * a * rho * one_minus_a ^ 2
        @. J["power", "rho"] = 2.0 * a_times_area * Vu ^ 3 * (one_minus_a)^2
        @. J["power", "Vu"] = 6.0 * Area * Vu^2 * a * rho * one_minus_a^2
end

actuator_disc_compute_partials2! = pyfunction(actuator_disc_compute_partials!,
                                              PyDict{String, PyAny},
                                              PyDict{String, PyArray},
                                              PyDict{Tuple{String, String}, PyArray})
options_data = []
input_data = [
              OM_Var_Data("a", 1, [0.5]),
              OM_Var_Data("Area", 1, [10.0]),
              OM_Var_Data("rho", 1, [1.225]),
              OM_Var_Data("Vu", 1, [10.0])
             ]
output_data = [OM_Var_Data("Vr", 1, [0.0]),
               OM_Var_Data("Vd", 1, [0.0]),
               OM_Var_Data("Ct", 1, [0.0]),
               OM_Var_Data("thrust", 1, [0.0]),
               OM_Var_Data("Cp", 1, [0.0]),
               OM_Var_Data("power", 1, [0.0])
              ]

partials_data = [
                 OM_Partials_Data("Vr", "a"),
                 OM_Partials_Data("Vr", "Vu"),
                 OM_Partials_Data("Vd", "a"),
                 OM_Partials_Data("Ct", "a"),
                 OM_Partials_Data("thrust", "a"),
                 OM_Partials_Data("thrust", "Area"),
                 OM_Partials_Data("thrust", "rho"),
                 OM_Partials_Data("thrust", "Vu"),
                 OM_Partials_Data("Cp", "a"),
                 OM_Partials_Data("power", "a"),
                 OM_Partials_Data("power", "Area"),
                 OM_Partials_Data("power", "rho"),
                 OM_Partials_Data("power", "Vu")
                ]

actuator_disc_data = OM_Ecomp_Data(actuator_disc_compute2!, actuator_disc_compute_partials2!, options_data, input_data, output_data, partials_data)

actuator_disc = julia_comps.JuliaExplicitComp(julia_comp_data=actuator_disc_data)

prob = om.Problem()

indeps = om.IndepVarComp()
indeps.add_output("a", 0.5)
indeps.add_output("Area", 10.0)
indeps.add_output("rho", 1.125)
indeps.add_output("Vu", 10.0)
prob.model.add_subsystem("indeps", indeps, promotes=["*"])

prob.model.add_subsystem("a_disc", actuator_disc, promotes_inputs=["a", "Area", "rho", "Vu"])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

prob.model.add_design_var("a", lower=0., upper=1.)
prob.model.add_design_var("Area", lower=0., upper=1.)

# negative one so we maximize the objective
prob.model.add_objective("a_disc.Cp", scaler=-1)

prob.setup()
prob.run_driver()

# minimum value
@show prob.get_val("a_disc.Cp")
@show prob.get_val("a")
