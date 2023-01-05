# [The Python-Centric Approach](@id python_centric)

## Installation
First step is to install `omjlcomps`, which is in the Python Package Index, so a simple

```bash
pip install omjlcomps
```

should be all you need.

## A Quick Tutorial
We're going to duplicate the [Paraboloid example from the OpenMDAO documentation](https://openmdao.org/newdocs/versions/latest/basic_user_guide/single_disciplinary_optimization/first_analysis.html), but implement the single `ExplicitComponent` in Julia instead of Python.
The goal of this tutorial is to minimize the paraboloid
```math
f(x,y) = (x - 3.0)^2 + xy + (y + 4.0)^2 - 3.0
```
with respect to ``x`` and ``y``.
The OpenMDAO docs say the answer is ``x = \frac{20}{3} \approx 6.667`` and ``y = -\frac{22}{3} \approx -7.333``.
Let's find out!

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
Like this, using `OpenMDAOCore.jl`:

```julia
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

Now we need to figure out how to get that Julia code into Python.
We'll use JuliaCall, provided by the [PythonCall](https://github.com/cjdoris/PythonCall.jl) package, for that.

```python
import openmdao.api as om

# Create a new Julia module that will hold all the Julia code imported into this Python module.
import juliacall; jl = juliacall.newmodule("ParaboloidExample")

# This assumes the file with the Julia Paraboloid implementation is in the current directory and is named `paraboloid.jl`.
jl.include("paraboloid.jl")
# Now we have access to everything in `paraboloid.jl`.

# Start creating an OpenMDAO Problem.
prob = om.Problem()

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

It worked, and we got the answer for ``x`` and ``y`` that we expected.
Yay!

The above Python run script should look pretty familiar if you have experience using OpenMDAO.
The Julia implementation of the Paraboloid will be less familiar, but we'll explain it step-by-step now.

## OpenMDAOCore.jl Explicit Components, Step-By-Step
Now let's go through each part of the Julia implementation of the paraboloid and figure out how it works.

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
The `VarData` and `PartialsData` docstrings have all the details:
```@docs
OpenMDAOCore.VarData
OpenMDAOCore.PartialsData
```

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

## A Fancier Example: A Nonlinear Circuit
The Paraboloid example used a few of the more basic features of OpenMDAO.jl.
Let's do something a bit more fancy: the [nonlinear circuit example](https://openmdao.org/newdocs/versions/latest/examples/circuit_analysis_examples.html) from the OpenMDAO documentation.
This example will include both explicit and implicit components, user-defined derivatives, and the OpenMDAO.jl approach to [component options](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/options.html).
This example uses three types of components: a `Resistor`, `Diode`, and `Node`.
Here's the implementation of the `Resistor`, first.

