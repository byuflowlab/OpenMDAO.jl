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
The OpenMDAO docs say the answer is ``x = \frac{20}{3}`` and ``y = -\frac{22}{3}``.
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
We'll use JuliaCall for that:

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
prob.setup()

prob.set_val('parab_comp.x', 3.0)
prob.set_val('parab_comp.y', -4.0)

prob.run_model()
print(prob['parab_comp.f_xy'])  # Should print `[-15.]`

prob.set_val('parab_comp.x', 5.0)
prob.set_val('parab_comp.y', -2.0)

prob.run_model()
print(prob.get_val('parab_comp.f_xy'))  # Should print `[-5.]`
```

Pretty nice!
