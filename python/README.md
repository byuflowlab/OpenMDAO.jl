# omjlcomps: OpenMDAO Julia Components

`omjlcomps` is a small Python package (actually, a [OpenMDAO Plugin](https://openmdao.org/newdocs/versions/latest/features/experimental/plugins.html)) that defines two classes, `JuliaExplicitComp` and `JuliaImplicitComp`, which inherit from OpenMDAO's `ExplicitComponent` and `ImplicitComponent`, respectively.
These components work with a Julia package called [OpenMDAOCore.jl](https://github.com/byuflowlab/OpenMDAO.jl) to create OpenMDAO `Components` that call Julia code.
Specifically, `JuliaExplicitComp` and `JuliaImplicitComp` take instances of concrete subtypes of `OpenMDAOCore.ExplicitComponent` and `OpenMDAOCore.ImplicitComponent` and turn them into instances of `JuliaExplicitComp` and `JuliaImplicitComp`.
Like any other OpenMDAO `ExplicitComponent` or `ImplicitComponent` objects, `JuliaExplicitComp` and `JuliaImplicitComp` instances can be used in an OpenMDAO model, but call Julia code in their methods (`compute`, `apply_nonlinear`, etc.).
See the [OpenMDAO.jl docs](http://flow.byu.edu/OpenMDAO.jl/dev/) for more information and examples.
