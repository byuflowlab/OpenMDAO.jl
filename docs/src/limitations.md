```@meta
CurrentModule = OpenMDAODocs
```
# Limitations

## Import `juliacall` first from Python... sometimes
When using the `omjlcomps` Python library, it is sometimes necessary to import `juliacall` before other Python libraries (at least `matplotlib`, maybe others too) to avoid an error that looks like this:

```
$ cat test.py
import matplotlib
import juliacall
$ python test.py
ERROR: `ccall` requires the compilerTraceback (most recent call last):
  File "/home/dingraha/desk/pythoncall_wtf/test.py", line 2, in <module>
    import juliacall
  File "/home/dingraha/desk/pythoncall_wtf/venv-mybuild-with-libc-enable-shared-without-lto-without-optimizations-computed-gotos-no-dtrace-no-ssl/lib/python3.9/site-packages/juliacall/__init__.py", line 218, in <module>
    init()
  File "/home/dingraha/desk/pythoncall_wtf/venv-mybuild-with-libc-enable-shared-without-lto-without-optimizations-computed-gotos-no-dtrace-no-ssl/lib/python3.9/site-packages/juliacall/__init__.py", line 214, in init
    raise Exception('PythonCall.jl did not start properly')
Exception: PythonCall.jl did not start properly
$
```

This only occurs when using the **system Python** on certain Linux distributions (e.g., Python 3.9.7 on Red Hat Enterprise Linux 8.6).
I've found three workarounds:

  * import the `juliacall` module first in your run script, before anything else, or
  * don't use the system Python: set up a Conda environment instead, or
  * don't use RHEL (the system Python on e.g. Arch Linux doesn't appear to suffer from this bug).

See [this PythonCall issue](https://github.com/cjdoris/PythonCall.jl/issues/255) for a few more details.

## Dymos ODEs and the truthiness of Julia callables in Python
To create a Dymos phase, the Dymos library requires the user to pass either a function or OpenMDAO `System` subclass that takes a `num_nodes` argument and return an OpenMDAO `System` that implements the ODE associated with the Dymos `Phase`.
How do we do that with OpenMDAO.jl?
It's easy enough to create, say, an `AbstractExplicitComp` that takes a `num_nodes` argument:

```@example dymos_callable
using OpenMDAOCore: OpenMDAOCore
struct FooODE <: OpenMDAOCore.AbstractExplicitComp
    num_nodes::Int
    # made-up options:
    a::Float64
    b::Bool
end
```

and then use `make_component` and a function to create a callable Julia function that fulfills Dymos' requirements:


```@example dymos_callable
using OpenMDAO: om, make_component
a = 8.0
b = true
make_ode(num_nodes) = make_component(FooODE(num_nodes, a, b))
println("make_ode = $(make_ode), typeof(make_ode) = $(typeof(make_ode))")
```

But can we use that to construct a Dymos `Phase`?

```@example dymos_callable
using PythonCall: pyimport
dm = pyimport("dymos")
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
phase = traj.add_phase("phase0",
                       dm.Phase(ode_class = make_ode,
                                transcription = dm.GaussLobatto(num_segments=10)))
```

Yes!
So we should be able to it in a trajectory optimization, right?

```@example dymos_callable
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
# p.setup() # Commented out to avoid error.
# This will throw something like:
#
# │   value =
# │    Python: TypeError: Julia: MethodError: no method matching length(::typeof(Main.__atexample__named__dymos_callable.make_ode))
# │    Closest candidates are:
# │      length(!Matched::Union{Base.KeySet, Base.ValueIterator}) at abstractdict.jl:58
# │      length(!Matched::Union{LinearAlgebra.Adjoint{T, S}, LinearAlgebra.Transpose{T, S}} where {T, S}) at ~/local/julia/1.8.5/share/julia/stdlib/v1.8/LinearAlgebra/src/adjtrans.jl:172
# │      length(!Matched::Union{Tables.AbstractColumns, Tables.AbstractRow}) at ~/.julia/packages/Tables/T7rHm/src/Tables.jl:180
# │      ...
```

Nope, sad: the `Problem.setup()` call on the last line of that example throws an error.
What's going wrong?
I am not 100% clear on all the details, but what I think is happening is this:

  * During the `Phase.setup` method, Dymos checks for the truthiness of the `ode_class` `Phase` `option`.
  * What is the truthiness of a class in Python?
    It depends:

      * If the `__bool__` method is defined for the class, that result of calling that is used,
      * If `__bool__` isn't defined, then if `__len__` is defined, the result of calling that is used, with `0` considered `False` and any other value `True`,
      * If neither `__bool__` nor `__len__` are defined, then the class is considered `True`.

  * The Julia function `make_ode` that we pass to the `Phase` constructor is converted to an `AnyValue` on the Python side by the Julia package PythonCall.
    The `AnyValue` class doesn't implement `__bool__`, but it **does** implement `__len__`, which calls the Julia function `length` on the Julia object.
  * But `length` doesn't have a method for plain old Julia functions, hence the error above.

The workaround is to create a small `struct` called [`DymosifiedCompWrapper`](@ref) that takes an `OpenMDAOCore.AbstractComp` type (not instance) and all the non-`num_nodes` arguments, and implement a `Base.length` method for `DymosifiedCompWrapper` that always returns `1`, a truthy value in Python.

See the [PythonCall issue on this topic](https://github.com/cjdoris/PythonCall.jl/issues/323) for news related to this topic.
