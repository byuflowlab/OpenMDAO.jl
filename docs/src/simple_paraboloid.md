```@meta
CurrentModule = OpenMDAODocs
```
# A Simple Example: Optimizing a Paraboloid
We're going to duplicate the [Paraboloid example from the OpenMDAO documentation](https://openmdao.org/newdocs/versions/latest/basic_user_guide/single_disciplinary_optimization/first_analysis.html), but implement the single `ExplicitComponent` in Julia instead of Python.
The goal of this tutorial is to minimize the paraboloid
```math
f(x,y) = (x - 3.0)^2 + xy + (y + 4.0)^2 - 3.0
```
with respect to ``x`` and ``y``.
The OpenMDAO docs say the answer is ``x = \frac{20}{3} \approx 6.667`` and ``y = -\frac{22}{3} \approx -7.333``.
Let's find out!

## The Python Implementation
One possible Python implementation of the above paraboloid is this:

```python
import openmdao.api as om


class Paraboloid(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Minimum at: x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0
```

Not too bad.
How do we do it in Julia?

## The Julia Implementation
Like this, using `OpenMDAOCore.jl`:

```@example paraboloid
using OpenMDAOCore: OpenMDAOCore

struct Paraboloid <: OpenMDAOCore.AbstractExplicitComp
end

function OpenMDAOCore.setup(self::Paraboloid)
    inputs = [OpenMDAOCore.VarData("x", val=0.0), OpenMDAOCore.VarData("y", val=0.0)]
    outputs = [OpenMDAOCore.VarData("f_xy", val=0.0)]
    partials = [OpenMDAOCore.PartialsData("*", "*", method="fd")]
    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::Paraboloid, inputs, outputs)
    x = inputs["x"][1]
    y = inputs["y"][1]

    outputs["f_xy"][1] = (x - 3.0)^2 + x * y + (y + 4.0)^2 - 3.0

    return nothing
end
```

What does all that mean?
We'll go through it step by step.

### Step 1: Preamble
```julia
using OpenMDAOCore: OpenMDAOCore
```

This line loads the OpenMDAOCore.jl Julia package.
Julia uses two different keywords for loading code from Julia modules: `using` and `import`.
The [official Julia docs on Modules](https://docs.julialang.org/en/v1/manual/modules/) do a good job of explaining the difference.
I like doing `using Foo: Foo` because it brings the module name `Foo` into the current scope, but not any of the names inside of `Foo`, so it doesn't clutter the namespace.
(The statement `using Foo: Foo` is kind of like Julia's version of `import foo` in Python, while just plain `using Foo` is like Python's `from foo import *`.)

### Step 2: The `Paraboloid` `struct`
```julia
struct Paraboloid <: OpenMDAOCore.AbstractExplicitComp
end
```
This bit of code defines a new type in Julia named `Paraboloid`.
The `<:` is the subtype operator in Julia, so we are telling Julia that our new `Paraboloid` type is a subtype of the `AbstractExplicitComp` type defined in `OpenMDAOCore`.
This is the Julian equivalent of
```python
class Paraboloid(om.ExplicitComponent):
```
in Python.

### Step 3: `OpenMDAOCore.setup`
```julia
function OpenMDAOCore.setup(self::Paraboloid)
    inputs = [OpenMDAOCore.VarData("x", val=0.0), OpenMDAOCore.VarData("y", val=0.0)]
    outputs = [OpenMDAOCore.VarData("f_xy", val=0.0)]
    partials = [OpenMDAOCore.PartialsData("*", "*", method="fd")]
    return inputs, outputs, partials
end
```
This `OpenMDAOCore.setup` method is the Julian equivalent of the `ExplicitComponent.setup` method from the Python version of the paraboloid.
The job of `OpenMDAOCore.setup` is to take a single argument (an `OpenMDAOCore.AbstractExplicitComp` or `OpenMDAOCore.AbstractImplicitComp`) and return three things:

  * A `Vector` of `VarData` `structs` containing metadata for the inputs to the component
  * A `Vector` of `VarData` `structs` containing metadata for the outputs of the component
  * A `Vector` of `PartialsData` `structs` containing metadata for the partial derivatives of the component

These Julia `Vector`s must always be returned in that order: inputs, outputs, partials.
OpenMDAO.jl uses the `VarData` entries in the `inputs` and `outputs` `Vectors` to construct arguments to the `Component.add_input` and `Component.add_output`, respectively.
And OpenMDAO.jl uses the `PartialsData` entries in the `partials` `Vector` to construct arguments to `Component.declare_partials`.
The [`OpenMDAOCore.VarData`](@ref) and [`OpenMDAOCore.PartialsData`](@ref) docstrings have all the details.

### Step 4: `OpenMDAOCore.compute!`
```julia
function OpenMDAOCore.compute!(self::Paraboloid, inputs, outputs)
    x = inputs["x"][1]
    y = inputs["y"][1]

    outputs["f_xy"][1] = (x - 3.0)^2 + x * y + (y + 4.0)^2 - 3.0

    return nothing
end
```
This `OpenMDAOCore.compute!` method is the equivalent of the `Paraboloid.compute` method from the Python version of the Paraboloid.
Its job is to take a `Paraboloid` `struct` and a `Dict` of inputs, calculate the outputs, and then store these outputs in the `outputs` `Dict`.
The `inputs` and `outputs` `Dict` entries are Julia arrays, similar to the NumPy arrays that OpenMDAO uses.
(They are actually [`PyArray`s](https://cjdoris.github.io/PythonCall.jl/stable/pythoncall-reference/#PythonCall.PyArray) from the `PythonCall` package, which are wrappers around the NumPy arrays that OpenMDAO creates for us.)

Now we need to figure out how to get that Julia code into OpenMDAO.
How we do that depends on whether we're following the Python-Centric Approach or Julia-Centric Approach.

### The Python-Centric Run Script
We'll use JuliaCall, provided by the [PythonCall](https://github.com/cjdoris/PythonCall.jl) package, to import the Julia code from the previous section into Python.
Then we can use the `omjlcomps` Python package to create an OpenMDAO `ExplicitComponent` from the `Paraboloid` Julia `struct`, and write a run script as usual.

```python
import openmdao.api as om

# Create a new Julia module that will hold all the Julia code imported into this Python module.
import juliacall; jl = juliacall.newmodule("ParaboloidExample")

# This assumes the file with the Julia Paraboloid implementation is in the current directory and is named `paraboloid.jl`.
jl.include("paraboloid.jl")
# Now we have access to everything in `paraboloid.jl`.

# omjlcomps knows how to create an OpenMDAO ExplicitComponent from an OpenMDAOCore.AbstractExplicitComp
from omjlcomps import JuliaExplicitComp
comp = JuliaExplicitComp(jlcomp=jl.Paraboloid())

# Now everything else is the same as https://openmdao.org/newdocs/versions/latest/basic_user_guide/single_disciplinary_optimization/first_analysis.html
model = om.Group()
model.add_subsystem('parab_comp', comp)

prob = om.Problem(model)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('parab_comp.x')
prob.model.add_design_var('parab_comp.y')
prob.model.add_objective('parab_comp.f_xy')

prob.setup()

prob.set_val('parab_comp.x', 3.0)
prob.set_val('parab_comp.y', -4.0)

prob.run_model()
print(prob['parab_comp.f_xy'])  # Should print `[-15.]`

prob.set_val('parab_comp.x', 5.0)
prob.set_val('parab_comp.y', -2.0)

prob.run_model()
print(prob.get_val('parab_comp.f_xy'))  # Should print `[-5.]`

prob.run_driver()
print(f"f_xy = {prob.get_val('parab_comp.f_xy')}")  # Should print `[-27.33333333]`
print(f"x = {prob.get_val('parab_comp.x')}")  # Should print `[6.66666633]`
print(f"y = {prob.get_val('parab_comp.y')}")  # Should print `[-7.33333367]`
```

The above Python run script should look pretty familiar if you have experience using OpenMDAO.
The only difference from a pure-Python version is the little bit at the top that we use to create the `JuliaExplicitComp`.

### The Julia-Centric Run Script
Now let's see if we can write a Julia run script:

```@example paraboloid
using OpenMDAO: om, make_component

prob = om.Problem()

# omjlcomps knows how to create an OpenMDAO ExplicitComponent from an OpenMDAOCore.AbstractExplicitComp
comp = make_component(Paraboloid())

model = om.Group()
model.add_subsystem("parab_comp", comp)

prob = om.Problem(model)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"

prob.model.add_design_var("parab_comp.x")
prob.model.add_design_var("parab_comp.y")
prob.model.add_objective("parab_comp.f_xy")

prob.setup()

prob.set_val("parab_comp.x", 3.0)
prob.set_val("parab_comp.y", -4.0)

prob.run_model()
println(prob["parab_comp.f_xy"])  # Should print `[-15.]`

prob.set_val("parab_comp.x", 5.0)
prob.set_val("parab_comp.y", -2.0)

prob.run_model()
println(prob.get_val("parab_comp.f_xy"))  # Should print `[-5.]`

prob.run_driver()
println("f_xy = $(prob.get_val("parab_comp.f_xy"))")  # Should print `[-27.33333333]`
println("x = $(prob.get_val("parab_comp.x"))")  # Should print `[6.66666633]`
println("y = $(prob.get_val("parab_comp.y"))")  # Should print `[-7.33333367]`
```

(This example assumes that the definition of the `Paraboloid` `struct` is included in the same file.
So concatenate those two code blocks if you'd like to run this yourself.)
Good news—we got the expected answer!

## Adding Derivatives
In the previous example we used OpenMDAO's finite difference method to approximate the paraboloid's partial derivatives.
We can calculate them ourselves, though, just like in a Python OpenMDAO Component.
Here's the implementation:

```@example paraboloid_up
using OpenMDAOCore: OpenMDAOCore

struct ParaboloidUserPartials <: OpenMDAOCore.AbstractExplicitComp
end

function OpenMDAOCore.setup(self::ParaboloidUserPartials)
    inputs = [OpenMDAOCore.VarData("x", val=0.0), OpenMDAOCore.VarData("y", val=0.0)]
    outputs = [OpenMDAOCore.VarData("f_xy", val=0.0)]
    partials = [OpenMDAOCore.PartialsData("*", "*")]
    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::ParaboloidUserPartials, inputs, outputs)
    x = inputs["x"][1]
    y = inputs["y"][1]

    outputs["f_xy"][1] = (x - 3.0)^2 + x * y + (y + 4.0)^2 - 3.0

    return nothing
end

function OpenMDAOCore.compute_partials!(self::ParaboloidUserPartials, inputs, partials)
    x = inputs["x"][1]
    y = inputs["y"][1]

    partials["f_xy", "x"][1] = 2*(x - 3.0) + y
    partials["f_xy", "y"][1] = x + 2*(y + 4.0)

    return nothing
end
```

The implementation of `ParaboloidUserPartials` is almost the same as `Paraboloid`.
The are only two differences:

  * We've removed the `method="fd"` argument from the call to the `PartialsData` constructor.
    This means the `method` argument will default to `"exact"` (as shown in the docstring above), and OpenMDAO will expect we'll calculate the derivatives of this component ourselves.
  * We've implemented a `compute_partials!` method for our new `ParaboloidUserPartials` `struct`.
    This is just like an `ExplicitComponent.compute_partials` method in a Python OpenMDAO component.
    Its job is to calculate the the derivatives of the outputs with respect to the inputs, of course.

So, we implemented a `compute_partials!` method.
But how do we know if they're right?
The OpenMDAO `Problem` class has a method called `check_partials` that compares the user-defined partial derivatives to the finite difference method.
Can we use that with an `OpenMDAOCore.AbstractExplicitComp`?
Let's try!

```@example paraboloid_up
using OpenMDAO: om, make_component

prob = om.Problem()

# omjlcomps knows how to create an OpenMDAO ExplicitComponent from an OpenMDAOCore.AbstractExplicitComp
comp = make_component(ParaboloidUserPartials())

model = om.Group()
model.add_subsystem("parab_comp", comp)

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
println(prob.check_partials(method="fd"))
```

It worked!
And the error is quite small.

What about the complex step method?

```@example paraboloid_up
println(prob.check_partials(method="cs"))
```

It works!
(The error is zero since the complex-step method is second-order accurate and we're differentiating a second-order polynomial.)
Complex numbers are no problem for Julia, but just like Python, we need to be careful to write our `compute_partials!` function in a complex-step-safe manner.

!!! note "FLOWMath.jl"
    The Julia library [FLOWMath](https://github.com/byuflowlab/FLOWMath.jl) has a collection of complex-step-safe functions.

Now, let's try an optimization:

```@example paraboloid_up
prob.run_driver()
println("f_xy = $(prob.get_val("parab_comp.f_xy"))")  # Should print `[-27.33333333]`
println("x = $(prob.get_val("parab_comp.x"))")  # Should print `[6.66666633]`
println("y = $(prob.get_val("parab_comp.y"))")  # Should print `[-7.33333367]`
```

Still works, and we got the right answer.
