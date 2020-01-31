from julia.OpenMDAO import (get_pysetup, get_pycompute, get_pycompute_partials,
                            get_pyapply_nonlinear, get_pylinearize,
                            get_pyguess_nonlinear, get_pysolve_nonlinear, make_component)


# these get methods are julia functions that create python 
# wrappers which avoid memory copying when passing the inputs 
# and outputs down to julia
setup = get_pysetup
compute = get_pycompute
compute_partials = get_pycompute_partials
apply_nonlinear = get_pyapply_nonlinear
# apply_linear = get_pyapply_linear
linearize = get_pylinearize
guess_nonlinear = get_pyguess_nonlinear
solve_nonlinear = get_pysolve_nonlinear