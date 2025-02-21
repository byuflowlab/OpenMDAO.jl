import juliacall
import openmdao.api as om
from omjlcomps import JuliaExplicitComp
from example_python_package_openmdao_jl.paraboloid import get_parabaloid_comp

prob = om.Problem()

model = om.Group()

jlcomp = get_parabaloid_comp()
parab_comp = JuliaExplicitComp(jlcomp=jlcomp)
model.add_subsystem("parab_comp", parab_comp)

prob = om.Problem(model)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

prob.model.add_design_var("parab_comp.x")
prob.model.add_design_var("parab_comp.y")
prob.model.add_objective("parab_comp.f_xy")

prob.setup(force_alloc_complex=True)

prob.set_val("parab_comp.x", 3.0)
prob.set_val("parab_comp.y", -4.0)

prob.run_model()
print(prob["parab_comp.f_xy"])  # Should print `[-15.]`

prob.set_val("parab_comp.x", 5.0)
prob.set_val("parab_comp.y", -2.0)

prob.run_model()
print(prob.get_val("parab_comp.f_xy"))  # Should print `[-5.]`

prob.check_partials(method="cs")

prob.run_driver()
print(f"f_xy = {prob.get_val('parab_comp.f_xy')} (should be -27.333333)")
print(f"x = {prob.get_val('parab_comp.x')} (should be 6.666666)")
print(f"y = {prob.get_val('parab_comp.y')} (should be 7.333333)")
