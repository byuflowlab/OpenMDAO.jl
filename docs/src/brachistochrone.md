```@meta
CurrentModule = OpenMDAODocs
```
# A Simple Dymos Example
We can also use OpenMDAO.jl Components within a Dymos ODE.
This example will implement [the Brachistochrone example from the Dymos docs](https://openmdao.github.io/dymos/examples/brachistochrone/brachistochrone.html) using Julia.

## Preamble
Let's make things easier on ourselves and import the `VarData` and `PartialsData` names into our local namespace:

```@example brachistochrone
using OpenMDAOCore: OpenMDAOCore, VarData, PartialsData
```

## The `AbstractExplicitComp` `struct`

The ODE component for the brachistochrone has two options:

  * `num_nodes`, the number of nodes used to descritize the trajectory of the bead.
  * `static_gravity`, a flag to indicate whether gravity should vary along the trajectory (and thus have length `num_nodes`) or if it should be constant (and thus be a scalar).

We'll use a default value of `false` for `static_gravity`, just like the Python implementation in the Dymos docs:

```@example brachistochrone
struct BrachistochroneODE <: OpenMDAOCore.AbstractExplicitComp
    num_nodes::Int
    static_gravity::Bool
end

# `static_gravity` set to `false` by default.
BrachistochroneODE(; num_nodes, static_gravity=false) = BrachistochroneODE(num_nodes, static_gravity)
```

## `OpenMDAOCore.setup`
Next we'll define the `OpenMDAOCore.setup` function:

```@example brachistochrone
function OpenMDAOCore.setup(self::BrachistochroneODE)
	nn = self.num_nodes

	# Inputs
    input_data = Vector{VarData}()
    push!(input_data, VarData("v"; val=zeros(nn), units="m/s"))
	if self.static_gravity
        push!(input_data, VarData("g", val=9.80665, units="m/s/s", tags=["dymos.static_target"]))
	else
        push!(input_data, VarData("g", val=9.80665 * ones(nn), units="m/s/s"))
    end
    push!(input_data, VarData("theta", val=ones(nn), units="rad"))

    # Outputs
    output_data = Vector{VarData}()
    push!(output_data, VarData("xdot", val=zeros(nn), units="m/s", tags=["dymos.state_rate_source:x", "dymos.state_units:m"]))
    push!(output_data, VarData("ydot", val=zeros(nn), units="m/s", tags=["dymos.state_rate_source:y", "dymos.state_units:m"]))
    push!(output_data, VarData("vdot", val=zeros(nn), units="m/s**2", tags=["dymos.state_rate_source:v", "dymos.state_units:m/s"]))
    push!(output_data, VarData("check", val=zeros(nn), units="m/s"))

    # Setup partials
    arange = 0:nn-1
    partials_data = Vector{PartialsData}()
    push!(partials_data, PartialsData("vdot", "theta"; rows=arange, cols=arange))

    push!(partials_data, PartialsData("xdot", "v"; rows=arange, cols=arange))
    push!(partials_data, PartialsData("xdot", "theta"; rows=arange, cols=arange))

    push!(partials_data, PartialsData("ydot", "v"; rows=arange, cols=arange))
    push!(partials_data, PartialsData("ydot", "theta"; rows=arange, cols=arange))

    push!(partials_data, PartialsData("check", "v"; rows=arange, cols=arange))
    push!(partials_data, PartialsData("check", "theta"; rows=arange, cols=arange))

	if self.static_gravity
		c = zeros(Int, self.num_nodes)
        push!(partials_data, PartialsData("vdot", "g"; rows=arange, cols=c))
	else
        push!(partials_data, PartialsData("vdot", "g"; rows=arange, cols=arange))
    end

    return input_data, output_data, partials_data
end
```

This is probably the most complicated `setup` we've seen yet.
A few things to note:

  * We can change the size of the `g` (gravity) input and its sub-Jacobian using the `static_gravity` option in the `BrachistochroneODE` `struct`.
  * The `VarData` calls use `tags`, which are passed to the `ExplicitComponent.add_input` method using the `tags` keyword argument in OpenMDAO.
  * As we'll see, the job of a Dymos ODE is to compute the state rates from the states and controls.
    It turns out that in many (all) ODEs, these state rates for a given trajectory node only depend on the state and controls at that particular node.
    This implies that the Jacobian of the ODE calculation will be sparse.
    This example, like the original Python implementation, passes the sparsity pattern of the various sub-Jacobians to the `PartialsData` structs using the `rows` and `cols` keywords.

## `OpenMDAOCore.compute!` and `OpenMDAOCore.compute_partials!`
The `OpenMDAOCore.compute!` method for our ODE is fairly straightforward:

```@example brachistochrone
function OpenMDAOCore.compute!(self::BrachistochroneODE, inputs, outputs)
	theta = inputs["theta"]
	cos_theta = cos.(theta)
	sin_theta = sin.(theta)
	g = inputs["g"]
	v = inputs["v"]

	@. outputs["vdot"] = g * cos_theta
	@. outputs["xdot"] = v * sin_theta
	@. outputs["ydot"] = -v * cos_theta
	@. outputs["check"] = v / sin_theta

    return nothing
end
```

The `@.` macro tells Julia to use [broadcasting](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting) for the array calculations (similar to NumPy broadcasting).

The `compute_partials!` method is also quite similar to the original Python implementation:

```@example brachistochrone
function OpenMDAOCore.compute_partials!(self::BrachistochroneODE, inputs, partials)
	theta = inputs["theta"]
	cos_theta = cos.(theta)
	sin_theta = sin.(theta)
	g = inputs["g"]
	v = inputs["v"]

	@. partials["vdot", "g"] = cos_theta
	@. partials["vdot", "theta"] = -g * sin_theta

	@. partials["xdot", "v"] = sin_theta
	@. partials["xdot", "theta"] = v * cos_theta

	@. partials["ydot", "v"] = -cos_theta
	@. partials["ydot", "theta"] = v * sin_theta

	@. partials["check", "v"] = 1 / sin_theta
	@. partials["check", "theta"] = -v * cos_theta / sin_theta ^ 2

	return nothing
end
```

## The run script
We'll need the Dymos library to solve the Brachistochrone problem, of course:

```@example brachistochrone
using PythonCall: pyimport
dm = pyimport("dymos")
```

And then the rest of the script will be pretty much identical to the Python version, but written in Julia.
We'll put it in a function that allows us to try out `static_gravity=false` and `static_gravity=true`.

```@example brachistochrone
using OpenMDAO: om, make_component  #, DymosifiedCompWrapper

function doit(; static_gravity)
  #
  # Initialize the Problem and the optimization driver
  #
  p = om.Problem(model=om.Group())
  p.driver = om.ScipyOptimizeDriver()
  p.driver.declare_coloring()
  #
  # Create a trajectory and add a phase to it
  #
  traj = p.model.add_subsystem("traj", dm.Trajectory())

  # `Trajectory.add_phase` expects a class that it can instantiate with the number of nodes used for the phase.
  # That's easy enough to create with an anonymous function, where we create a function that takes the single keyword argument `num_nodes`, use that to create a `BrachistochroneODE` with the appropriate value for `static_gravity`, and then pass that to the `make_component` function that returns an OpenMDAO `Component`.
  phase = traj.add_phase("phase0",
                         dm.Phase(ode_class = (; num_nodes)->make_component(BrachistochroneODE(; num_nodes=num_nodes, static_gravity=static_gravity)),
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

doit(; static_gravity=false)
doit(; static_gravity=true)
```

At the end we see we got pretty much the same answer for the elapsed time as the Python example in the Dymos docs.
(But not exactly the same, which is a bummer...)
