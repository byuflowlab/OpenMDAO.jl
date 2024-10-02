```@meta
CurrentModule = OpenMDAODocs
```

# (Experimental) Automatic Sparse Forward-Mode AD
OpenMDAOCore.jl can create explicit components that are differentiated automatically using the [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) package, with the derivative calculation accelerated by [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl).
OpenMDAOCore.jl will automatically determine the sparsity pattern of a given component's Jacobian, and then functions from SparseDiffTools.jl will use a coloring algorithm to color the Jacobian, and use the coloring information to accelerate the derivative calculation.
The sparsity pattern of the component will also be communicated to OpenMDAO via the `rows` and `cols` arguments to `add_input` and `add_output`.

## The User-Defined Function
The interface for the automatic sparse forward-mode AD functionality is a bit different from the "plain" `AbstractExplicitComp` and `AbstractImplicitComp` `struct`s described in earlier examples (see [A Simple Example: Optimizing a Paraboloid](@ref) or [A More Complicated Example: Nonlinear Circuit](@ref)).
Instead of creating subtypes of `AbstractExplicitComp` that implement `OpenMDAOCore.setup`, `OpenMDAOCore.compute!`, etc., we'll be writing a Julia function that performs our desired computation.
This user-defined function will then be passed to a constructor of the `SimpleAutoSparseForwardDiffExplicitComp` `struct`, which will implement the necessary `OpenMDAOCore` methods for us.

The user-defined function *must* follow a particular form: it must take exactly three arguments:

  * A `ComponentVector` `Y_ca` that contains the outputs of the explicit component,
  * A `ComponentVector` `X_ca` that contains the inputs of the explicit component,
  * `params`, which holds any parameters needed for the component's computation that will not be differentiated,

in that order.
In other words, the function should look like this:

```julia
function my_function!(Y_ca, X_ca, params)
     # Use X_ca and optionally params to calculate outputs and write them to Y_ca

     return nothing
end
```

(The function doesn't have to return nothing, but any returned value will be ignored, so I like to include `return nothing` to make it clear that the return value doesn't matter.)

`X_ca` and `Y_ca` need to be `ComponentVectors`.
What are those?
They are objects provided by the [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl) package that act like `Vector`s, but allow the user to define names for each part ("component") of the vector.
For example:

```@example component_vectors
using ComponentArrays: ComponentVector

x1 = ComponentVector(foo=-1.0, bar=-2.0, baz=-3.0)
@show x1 x1[3] x1.foo x1[:foo]
nothing # hide
```

Notice that we can get, say, the third value of `x1` the usual way (`x1[3]`), but also by referring to the `foo` field value via `x1.foo` and by indexing the `ComponentVector` with the symbol `:foo` (`x1[:foo]`).

Each of the components in `x1` are scalars, but they don't have to be:

```@example component_vectors
x2 = ComponentVector(foo=-1.0, bar=1:4, baz=reshape(5:10, 2, 3))
@show x2 x2[:foo] x2[:bar] x2[:baz]
nothing # hide
```

In `x2`, the `foo` component is a scalar, `bar` refers to a `Vector` (aka a 1D `Array`) and `baz` refers to a `Matrix` (aka a 2D Array).
But `x2` still "looks like" a `Vector`:

```@example component_vectors
@show x2[3]  # will give the third value of `x2`, which happens to be the second value of x2[:bar]
@show ndims(x2)  # Should be 1, since a Vector is 1-dimensional
@show length(x2)  # length(x2) gives the total number of entries in `x2`, aka 1 + 4 + 2*3 = 11
@show size(x2)  # size is a length-1 tuple since a Vector has just one dimension
nothing # hide
```

Now, how will we use `ComponentVector`s here?
We'll use them to define the names and sizes of all the inputs and outputs to our component.
For example, with the paraboloid component in [A Simple Example: Optimizing a Paraboloid](@ref), we created one component with two inputs `x` and `y` and one output `f_xy`, all scalars.
So for that case, our `X_ca` would be

```@example component_vectors
X_ca = ComponentVector(x=1.0, y=1.0)
Y_ca = ComponentVector(f_xy=0.0)
@show X_ca Y_ca
nothing # hide
```

Actually, why don't we try to implement the `Paraboloid` component using a `SimpleAutoSparseForwardDiffExplicitComp`?

## `SimpleAutoSparseForwardDiffExplicitComp` Paraboloid 
We'll start fresh, first with importing the stuff we'll need:

```@example simple_auto_sparse_forwarddiff_paraboloid
using ComponentArrays: ComponentVector
using OpenMDAOCore: OpenMDAOCore
```

Next, we need to define the function that implements our paraboloid equation, which, again, is

```math
f(x,y) = (x - 3.0)^2 + xy + (y + 4.0)^2 - 3.0
```

That would look like this:

```@example simple_auto_sparse_forwarddiff_paraboloid
function f_paraboloid!(Y_ca, X_ca, params)
    # Get the inputs:
    x = @view(X_ca[:x])
    y = @view(X_ca[:y])

    # Get the output:
    f_xy = @view(Y_ca[:f_xy])

    # Do the calculation:
    @. f_xy = (x - 3.0)^2 + x*y + (y + 4.0)^2 - 3.0

    # Return value doesn't matter.
    return nothing
end
nothing # hide
```

A couple of things to note there:

  * The `@view` macro is used when extracting the inputs and outputs from the `X_ca` and `Y_ca` `ComponentVector`s.
    This creates a view into the original `ComponentVector`, instead of a new array with a copy of the original data, which avoids unnecessary allocations and (for the outputs) allows modifications to the view to be reflected in the `Y_ca` array.
    In this example everything is a scalar, so no allocations would have happened anyway.
    But it doesn't hurt to use `@view`: it's a good habit to get into, *and* it allows us to use the `@.` [broadcasting](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting) macro with the scalar `f_xy` output.
  * The `params` argument is not used in this example, but it is still required, since the `SimpleAutoSparseForwardDiffExplicitComp` constructor will expect the function to accept it.

Our next step is to create the `ComponentVector`s that will be used to hold the inputs and outputs:

```@example simple_auto_sparse_forwarddiff_paraboloid
X_ca = ComponentVector(x=1.0, y=1.0)
Y_ca = ComponentVector(f_xy=0.0)
@show X_ca Y_ca
nothing # hide
```

!!! warning "Use sane values for `X_ca`"

    The values of the entries in `X_ca` will be perturbed slightly when detecting the sparsity pattern of the component's Jacobian, so don't use nonsense values or `NaN`!


Now we're ready to create the `SimpleAutoSparseForwardDiffExplicitComp`:

```@example simple_auto_sparse_forwarddiff_paraboloid
using OpenMDAO: make_component

comp = OpenMDAOCore.SimpleAutoSparseForwardDiffExplicitComp(f_paraboloid!, Y_ca, X_ca)
parab_comp = make_component(comp)
nothing # hide
```

`make_component` will convert the `SimpleAutoSparseForwardDiffExplicitComp` into a OpenMDAO Python component that we can use with OpenMDAO.
So now we just need to proceed with the paraboloid example as usual:

```@example simple_auto_sparse_forwarddiff_paraboloid
using OpenMDAO: om

prob = om.Problem()

model = om.Group()
model.add_subsystem("parab_comp", parab_comp)

prob = om.Problem(model)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

prob.model.add_design_var("parab_comp.x")
prob.model.add_design_var("parab_comp.y")
prob.model.add_objective("parab_comp.f_xy")

prob.setup(force_alloc_complex=true)

prob.set_val("parab_comp.x", 3.0)
prob.set_val("parab_comp.y", -4.0)

prob.run_model()
println(prob["parab_comp.f_xy"])  # Should print `[-15.]`

prob.set_val("parab_comp.x", 5.0)
prob.set_val("parab_comp.y", -2.0)

prob.run_model()
println(prob.get_val("parab_comp.f_xy"))  # Should print `[-5.]`
nothing # hide
```

Looks OK so far.
But we should check our derivatives, just to be safe.
We can do that with the finite difference method:

```@example simple_auto_sparse_forwarddiff_paraboloid
println(prob.check_partials(method="fd"))
nothing # hide
```

or the complex-step method:

```@example simple_auto_sparse_forwarddiff_paraboloid
println(prob.check_partials(method="cs"))
nothing # hide
```

Derivatives look great, so let's go ahead and perform the optimization:

```@example simple_auto_sparse_forwarddiff_paraboloid
prob.run_driver()
println("f_xy = $(prob.get_val("parab_comp.f_xy"))")  # Should print `[-27.33333333]`
println("x = $(prob.get_val("parab_comp.x"))")  # Should print `[6.66666633]`
println("y = $(prob.get_val("parab_comp.y"))")  # Should print `[-7.33333367]`
nothing # hide
```

Victory!

## `SimpleAutoSparseForwardDiffExplicitComp` with Actual Sparsity
The Paraboloid example does a nice job of showing how to use `SimpleAutoSparseForwardDiffExplicitComp`, but careful readers will realize that it's not actually sparse!
So let's try an example with some sparse components: the [Simple Optimization using Simultaneous Derivatives](https://openmdao.org/newdocs/versions/latest/examples/simul_deriv_example.html) example from the OpenMDAO docs.
That example makes heavy use of `ExecComps` with `has_diag_partials`, but we'll use `SimpleAutoSparseForwardDiffExplicitComp` to implement each component, and allow it to find the sparsity pattern for us.

Let's start with a fresh Julia script and load what we'll need:

```@example simple_auto_sparse_forwarddiff_circle
using ComponentArrays: ComponentVector
using OpenMDAOCore: OpenMDAOCore
using OpenMDAO: om, make_component

# Use the same value of `SIZE` that the official version uses:
SIZE = 10
```

Now we'll need to create some functions that will implement the various calculations we need.
Here are the functions for `arctan_yox`, `circle`, and `r_con`, respectively:

```@example simple_auto_sparse_forwarddiff_circle
function f_arctan_yox!(Y, X, params)
    # Get views of inputs.
    x = @view(X[:x])
    y = @view(X[:y])

    # Get views of outputs.
    g = @view(Y[:g])

    # Perform the calculation.
    @. g = atan(y/x)

    return nothing
end
Y_ca = ComponentVector(g=ones(SIZE))
X_ca = ComponentVector(x=ones(SIZE), y=ones(SIZE))
comp = OpenMDAOCore.SimpleAutoSparseForwardDiffExplicitComp(f_arctan_yox!, Y_ca, X_ca)
arctan_yox_comp = make_component(comp)

function f_circle!(Y, X, params)
    # Get views of inputs.
    r = @view(X[:r])

    # Get views of outputs.
    area = @view(Y[:area])

    # Perform the calculation.
    @. area = pi*r^2

    return nothing
end
Y_ca = ComponentVector(area=1.0)
X_ca = ComponentVector(r=1.0)
comp = OpenMDAOCore.SimpleAutoSparseForwardDiffExplicitComp(f_circle!, Y_ca, X_ca)
circle_comp = make_component(comp)

function f_r_con!(Y, X, params)
    # Get views of inputs.
    x = @view(X[:x])
    y = @view(X[:y])
    r = @view(X[:r])

    # Get views of outputs.
    g = @view(Y[:g])

    # Perform the calculation.
    @. g = x^2 + y^2 - r

    return nothing
end
Y_ca = ComponentVector(g=ones(SIZE))
X_ca = ComponentVector(x=ones(SIZE), y=ones(SIZE), r=1.0)
comp = OpenMDAOCore.SimpleAutoSparseForwardDiffExplicitComp(f_r_con!, Y_ca, X_ca)
r_con_comp = make_component(comp)

nothing # hide
```

We can check that `SimpleAutoSparseForwardDiffExplicitComp` has detected the diagonal partials by getting the rows and columns `Dict` that it constructs.
For that last component:

```@example simple_auto_sparse_forwarddiff_circle
rcdict = OpenMDAOCore.get_rows_cols_dict(comp)
@show rcdict[:g, :x] rcdict[:g, :y] rcdict[:g, :r]
nothing # hide
```

The `rcdict` maps the partials to a tuple of two vectors containing the non-zero rows and non-zero columns of the partial.
For example, that output above shows that the derivative of 

  * `g[1]` with respect to `x[1]`
  * `g[2]` with respect to `x[2]`
  * `g[3]` with respect to `x[3]`

etc are all non-zero.

Next, let's think about the calculation for the `theta_con` component, which in the Python example looks like this:

```python
thetas = np.linspace(0, np.pi/4, SIZE)
p.model.add_subsystem('theta_con', om.ExecComp('g = x - theta', has_diag_partials=True,
                                               g=np.ones(SIZE), x=np.ones(SIZE),
                                               theta=thetas))
```

The `theta` array is hard-coded to go from `0` to `py/4`, but what if we wanted to be able to change that?
We could use the `params` argument to the user-defined function to set the starting and ending value for `theta`, like so:

```@example simple_auto_sparse_forwarddiff_circle
function f_theta_con!(Y, X, params)
    # Get views of inputs.
    x = @view(X[:x])

    # Get views of outputs.
    g = @view(Y[:g])

    # Create the theta parameter.
    theta_min = params.theta_min
    theta_max = params.theta_max
    theta = range(theta_min, theta_max; length=length(x))

    # Perform the calculation.
    @. g = x - theta

    return nothing
end

Y_ca = ComponentVector(g=ones(SIZE))
X_ca = ComponentVector(x=ones(SIZE))
params_theta_con = (theta_min=0.0, theta_max=pi/4)
comp = OpenMDAOCore.SimpleAutoSparseForwardDiffExplicitComp(f_theta_con!, Y_ca, X_ca; params=params_theta_con)
theta_con_comp = make_component(comp)
nothing # hide
```

Notice that we decided to use a `NamedTuple` for the `params` argument.
It could be anything, as long as it's consistent with the definition of `f_theta_con!`.
For example, we could have passed a `Vector` `[0.0, pi/4]` to `SimpleAutoSparseForwardDiffExplicitComp`, then done something like

```julia
theta_min = params[1]
theta_max = params[2]
```

in the definition of `f_theta_con!`.

Finally, let's create the functions for the `delta_theta_con` and `l_conx` components:

```@example simple_auto_sparse_forwarddiff_circle
function f_delta_theta_con!(Y, X, params)
    # Get views of inputs.
    even = @view(X[:even])
    odd = @view(X[:odd])

    # Get views of outputs.
    g = @view(Y[:g])

    # Perform the calculation.
    @. g = even - odd

    return nothing
end
Y_ca = ComponentVector(g=ones(SIZE÷2))
X_ca = ComponentVector(even=ones(SIZE÷2), odd=ones(SIZE÷2))
comp = OpenMDAOCore.SimpleAutoSparseForwardDiffExplicitComp(f_delta_theta_con!, Y_ca, X_ca)
delta_theta_con_comp = make_component(comp)

function f_l_conx!(Y, X, params)
    # Get views of inputs.
    x = @view(X[:x])

    # Get views of outputs.
    g = @view(Y[:g])

    # Perform the calculation.
    @. g = x - 1

    return nothing
end
Y_ca = ComponentVector(g=ones(SIZE))
X_ca = ComponentVector(x=ones(SIZE))
comp = OpenMDAOCore.SimpleAutoSparseForwardDiffExplicitComp(f_l_conx!, Y_ca, X_ca)
l_conx_comp = make_component(comp)

nothing # hide
```

Excellent.
Now we're finally ready to start creating the OpenMDAO `Problem`.
The rest of the example will closely follow the [official version](https://openmdao.org/newdocs/versions/latest/examples/simul_deriv_example.html).

```@example simple_auto_sparse_forwarddiff_circle

p = om.Problem()

p.model.add_subsystem("arctan_yox", arctan_yox_comp, promotes_inputs=["x", "y"])
p.model.add_subsystem("circle", circle_comp, promotes_inputs=["r"])
p.model.add_subsystem("r_con", r_con_comp, promotes_inputs=["r", "x", "y"])
p.model.add_subsystem("theta_con", theta_con_comp)
p.model.add_subsystem("delta_theta_con", delta_theta_con_comp)
p.model.add_subsystem("l_conx", l_conx_comp, promotes_inputs=["x"])

# OpenMDAO uses 0-based indices, so need to go from 0 to SIZE-1.
IND = 0:SIZE-1
# Julia arrays are 1-based, so 1 will give us the first index, 2 the second, etc..
# Could also use the popular OffsetArrays.jl package to create a 0-based array.
ODD_IND = IND[2:2:end]  # all odd indicies
EVEN_IND = IND[1:2:end-1]  # all even indices

p.model.connect("arctan_yox.g", "theta_con.x")
p.model.connect("arctan_yox.g", "delta_theta_con.even", src_indices=EVEN_IND)
p.model.connect("arctan_yox.g", "delta_theta_con.odd", src_indices=ODD_IND)

p.driver = om.ScipyOptimizeDriver()
p.driver.options["optimizer"] = "SLSQP"
p.driver.options["disp"] = false

# set up dynamic total coloring here
p.driver.declare_coloring()

p.model.add_design_var("x")
p.model.add_design_var("y")
p.model.add_design_var("r", lower=.5, upper=10)

# nonlinear constraints
p.model.add_constraint("r_con.g", equals=0)

p.model.add_constraint("theta_con.g", lower=-1e-5, upper=1e-5, indices=EVEN_IND)
p.model.add_constraint("delta_theta_con.g", lower=-1e-5, upper=1e-5)

# this constrains x[0] to be 1 (see definition of l_conx)
p.model.add_constraint("l_conx.g", equals=0, linear=false, indices=[0,])

# linear constraint
p.model.add_constraint("y", equals=0, indices=[0,], linear=true)

p.model.add_objective("circle.area", ref=-1)

p.setup(mode="fwd", force_alloc_complex=true)

# the following were randomly generated using np.random.random(10)*2-1 to randomly
# disperse them within a unit circle centered at the origin.
p.set_val("x", [ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                 0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886])
p.set_val("y", [ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582])
p.set_val("r", .7)

nothing # hide
```

Let's check our derivatives:

```@example simple_auto_sparse_forwarddiff_circle
println(p.check_partials(method="cs"))
nothing # hide
```

Essentially zero error in the derivatives, so that's great.

Finally, let's do the optimization:

```@example simple_auto_sparse_forwarddiff_circle
p.run_driver()
println("circle.area = $(p["circle.area"]) (should be about equal to π ≈ $(float(pi)))")
nothing # hide
```

We got the right answer!
Yay!

## The Brachistochrone with `SimpleAutoSparseForwardDiffExplicitComp`
Finally, let's implement the Brachistochrone problem from [A Simple Dymos Example](@ref) using `SimpleAutoSparseForwardDiffExplicitComp`.
This will demonstrate how to add units and tags to variables used in a `SimpleAutoSparseForwardDiffExplicitComp` component.

First, let's start a new Julia script, and import Dymos:

```@example simple_auto_sparse_forwarddiff_brachistochrone
using ComponentArrays: ComponentVector
using OpenMDAOCore: OpenMDAOCore
using OpenMDAO: om, make_component
using PythonCall: pyimport, pydict

dm = pyimport("dymos")
nothing # hide
```

Next, let's define the Brachistochrone ode:

```@example simple_auto_sparse_forwarddiff_brachistochrone
function brachistochrone_ode!(Y, X, params)
    theta = @view X[:theta]
    g = @view X[:g]
    v = @view X[:v]

    cos_theta = cos.(theta)
    sin_theta = sin.(theta)

    @view(Y[:vdot]) .= g .* cos_theta
    @view(Y[:xdot]) .= v .* sin_theta
    @view(Y[:ydot]) .= -v .* cos_theta
    @view(Y[:check]) .= v ./ sin_theta

    return nothing
end
nothing # hide
```

Now, for the tricky part: we need to give Dymos a function that, when passed the `num_nodes` keyword argument, returns a component that implements the ODE.
The component this yet-to-be-written function will return will be a `SimpleAutoSparseForwardDiffExplicitComp`, and that means we will need to also create the required `ComponentVector`s for the inputs and outputs.
How will we do that?
Like this:

```@example simple_auto_sparse_forwarddiff_brachistochrone
function brachistochrone_ode_factory(; num_nodes, static_gravity)
    v = rand(Float64, num_nodes) .+ 1
    theta = rand(Float64, num_nodes) .+ 2
    if static_gravity
        g = 9.81
    else
        g = fill(9.81, num_nodes)
    end
    X_ca = ComponentVector(v=v, theta=theta, g=g)

    xdot = zeros(num_nodes)
    ydot = zeros(num_nodes)
    vdot = zeros(num_nodes)
    check = zeros(num_nodes)
    Y_ca = ComponentVector(xdot=xdot, ydot=ydot, vdot=vdot, check=check)

    units_dict = Dict(:v=>"m/s", :theta=>"rad", :g=>"m/s**2", :xdot=>"m/s", :ydot=>"m/s", :vdot=>"m/s**2", :check=>"m/s")
    tags_dict = Dict(
        :xdot=>["dymos.state_rate_source:x", "dymos.state_units:m"],
        :ydot=>["dymos.state_rate_source:y", "dymos.state_units:m"],
        :vdot=>["dymos.state_rate_source:v", "dymos.state_units:m/s"],
    )
    if static_gravity
        tags_dict[:g] = ["dymos.static_target"]
    end

    return make_component(OpenMDAOCore.SimpleAutoSparseForwardDiffExplicitComp(brachistochrone_ode!, Y_ca, X_ca; units_dict=units_dict, tags_dict=tags_dict))
end
nothing # hide
```

`brachistochrone_ode_factory` takes to keyword arguments: the `num_nodes` argument that Dymos requires, and the `static_gravity` argument from the [original Dymos example](https://openmdao.github.io/dymos/examples/brachistochrone/brachistochrone.html).
Once we know the value of `num_nodes`, we can create the `X_ca` and `Y_ca` `ComponentVector`s as usual, with the `static_gravity` parameter influencing the size of `g`.
We also create two dictionaries: `units_dict` and `tags_dict`.
`units_dict` maps variable names (expressed as `Symbol`s, just like the component names for the `ComponentVector`s) to `String`s defining the units of the variable.
The `tags_dict` similarly maps `Symbol` variable names to `Vector`s of `String` specifying the tags for each variable.
Both `Dict`s are passed to the `SimpleAutoSparseForwardDiffExplicitComp` constructor as keyword arguments.
Finally, the resulting `SimpleAutoSparseForwardDiffExplicitComp` is passed to `make_component` to actually create the Python OpenMDAO component.

Now, let's actually define the problem and run the optimization.
We'll define a driver function called `doit` that allows us to try the example with the `static_gravity` argument set to `true` and `false`.

```@example simple_auto_sparse_forwarddiff_brachistochrone
function doit(; static_gravity)
    # Initialize the Problem and the optimization driver
    #
    p = om.Problem(model=om.Group())
    p.driver = om.ScipyOptimizeDriver()
    p.driver.declare_coloring()
    #
    # Create a trajectory and add a phase to it
    #
    traj = p.model.add_subsystem("traj", dm.Trajectory())
           

    phase = traj.add_phase("phase0",
                           dm.Phase(ode_class=brachistochrone_ode_factory, ode_init_kwargs=pydict(Dict("static_gravity"=>static_gravity)),
                                    transcription = dm.GaussLobatto(num_segments=10)))

    #
    # Set the variables
    #
    phase.set_time_options(fix_initial=true, duration_bounds=(.5, 10))

    phase.add_state("x", fix_initial=true, fix_final=true)

    phase.add_state("y", fix_initial=true, fix_final=true)

    phase.add_state("v", fix_initial=true, fix_final=false)

    phase.add_control("theta", continuity=true, rate_continuity=true,
                      units="deg", lower=0.01, upper=179.9)

    phase.add_parameter("g", units="m/s**2", val=9.80665)

    #
    # Minimize time at the end of the phase
    #
    phase.add_objective("time", loc="final", scaler=10)
    # 
    p.model.linear_solver = om.DirectSolver()
    #
    # Setup the Problem
    #
    p.setup()

    #
    # Set the initial values
    #
    p["traj.phase0.t_initial"] = 0.0
    p["traj.phase0.t_duration"] = 2.0

    p.set_val("traj.phase0.states:x", phase.interp("x", ys=[0, 10]))
    p.set_val("traj.phase0.states:y", phase.interp("y", ys=[10, 5]))
    p.set_val("traj.phase0.states:v", phase.interp("v", ys=[0, 9.9]))
    p.set_val("traj.phase0.controls:theta", phase.interp("theta", ys=[5, 100.5]))

    #
    # Solve for the optimal trajectory
    #
    dm.run_problem(p)

    # Check the results
    println("static_gravity = $static_gravity, elapsed time = $(p.get_val("traj.phase0.timeseries.time")[-1]) (should be 1.80164719)")
end

nothing # hide
```

The `doit` function is essentially identical to the previous version from [A Simple Dymos Example](@ref).
The only difference is in the `ode_class` argument in the call to `dm.Phase`: we assign that argument to our `brachistochrone_ode_factory` function.
We also need to tell Dymos about the value of the `static_gravity` option we would like to use, which can be done via the `ode_init_kwargs` argument.
In Python, this would normally be a `dict` with string keys.
To do this in Julia, we first create a `Dict` with `String` keys, then use the `pydict` function from PythonCall to convert the Julia `Dict` to a Python `dict`.

With all that in place, we can run the optimization!
Let's try with `static_gravity=true`:

```@example simple_auto_sparse_forwarddiff_brachistochrone
doit(; static_gravity=true)

nothing # hide
```

Got the right answer!

And now, `static_gravity=false`:

```@example simple_auto_sparse_forwarddiff_brachistochrone
doit(; static_gravity=false)

nothing # hide
```

Also looks good!
