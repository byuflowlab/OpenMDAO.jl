import openmdao.api as om
import julia.Main as julia
from julia.OpenMDAO import make_component


julia.include("components/actuator_disc.jl")

prob = om.Problem()

indeps = om.IndepVarComp()
indeps.add_output("a", 0.5)
indeps.add_output("Area", 10.0)
indeps.add_output("rho", 1.125)
indeps.add_output("Vu", 10.0)
prob.model.add_subsystem("indeps", indeps, promotes=["*"])

comp = make_component(julia.ActuatorDisc())
prob.model.add_subsystem("a_disc", comp, promotes_inputs=["a", "Area", "rho", "Vu"])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")

prob.model.add_design_var("a", lower=0., upper=1.)
prob.model.add_design_var("Area", lower=0., upper=1.)

# negative one so we maximize the objective
prob.model.add_objective("a_disc.Cp", scaler=-1)

prob.setup()
prob.run_driver()

# minimum value
print(f"a_disc.Cp = {prob.get_val('a_disc.Cp')} (should be 0.59259259)")
print(f"a = {prob.get_val('a')} (should be 0.33335528)")
