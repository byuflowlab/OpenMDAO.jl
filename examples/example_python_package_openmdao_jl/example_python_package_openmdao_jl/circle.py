import os

# Create a new Julia module that will hold all the Julia code imported into this Python module.
import juliacall; jl = juliacall.newmodule("CircleComponentsStub")

# Get the directory this file is in, then include the `circle.jl` Julia source code.
d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "circle.jl"))
# Now we have access to everything in `circle.jl` in the `jl` object.

get_arctan_yox_comp = jl.get_arctan_yox_comp
get_circle_comp = jl.get_circle_comp
get_r_con_comp = jl.get_r_con_comp
get_theta_con_comp = jl.get_theta_con_comp
get_delta_theta_con_comp = jl.get_delta_theta_con_comp
get_l_conx_comp = jl.get_l_conx_comp
