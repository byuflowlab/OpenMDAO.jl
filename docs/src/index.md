```@meta
CurrentModule = OpenMDAODocs
```
# OpenMDAO.jl Documentation

## What?
Use Julia with [OpenMDAO](https://openmdao.org/)!
OpenMDAO.jl is a Julia package that allows a user to:

  * Write OpenMDAO `Component`s in Julia, and incorporate these components into a OpenMDAO model.
  * Create and run optimizations in Julia, using OpenMDAO as a library.

OpenMDAO.jl consists of three pieces of software:

  * OpenMDAOCore.jl: A small, pure-Julia package that allows users to define Julia code that will eventually be used in an OpenMDAO `Problem`. OpenMDAOCore.jl defines two Julia abstract types (`AbstractExplicitComponent` and `AbstractImplicitComponent`) and methods that mimic OpenMDAO's ExplicitComponent and ImplicitComponent classes.
  * `omjlcomps`: A Python package (actually, a [OpenMDAO Plugin](https://openmdao.org/newdocs/versions/latest/features/experimental/plugins.html)) that defines two classes, `JuliaExplicitComp` and `JuliaImplicitComp`, which inherit from OpenMDAO's `ExplicitComponent` and `ImplicitComponent`, respectively.
    These components take instances of concrete subtypes of `OpenMDAOCore.ExplicitComponent` and `OpenMDAOCore.ImplicitComponent` and turn them into instances of `JuliaExplicitComp` and `JuliaImplicitComp`.
    Like any other OpenMDAO `ExplicitComponent` or `ImplicitComponent` objects, `JuliaExplicitComp` and `JuliaImplicitComp` instances can be used in an OpenMDAO model, but call Julia code in their methods (`compute`, `apply_nonlinear`, etc.).
  * OpenMDAO.jl: A Julia package that has the openmdao and `omjlcomps` Python packages as dependencies.
    Users can install `OpenMDAO.jl` and have the full power of the OpenMDAO framework at their disposal in Julia.

## How (Installation Instructions)?
There are two approaches to getting OpenMDAO working with Julia: the [Python-Centric Approach](@ref python_centric) and the [Julia-Centric Approach](@ref julia_centric).
If you like Python and just want to have a little (or a lot) of Julia buried in your OpenMDAO `System`, then you'll probably prefer the Python-centric approach.
If you're a huge fan of Julia and would like to pretend that OpenMDAO is a Julia library, you'll want the Julia-centric approach.
Either way, pick one or the other: you don't need to follow both installation instructions.

### [Python-Centric Installation](@id python_centric)
The first (and only!) step is to install `omjlcomps`, which is in the Python Package Index, so a simple

```bash
pip install omjlcomps
```

should be all you need.
`omjlcomps` uses [`JuliaPkg`](https://github.com/cjdoris/PyJuliaPkg) to manage Julia dependencies, so all the Julia packages needed by `omjlcomps` (and even Julia itself, if necessary) will be installed automatically.

### [Julia-Centric Installation](@id julia_centric)
The OpenMDAOCore.jl and OpenMDAO.jl Julia packages are registered in the [General registry](https://github.com/JuliaRegistries/General), so installation should be as simple as
```
] add OpenMDAOCore OpenMDAO
```
in the Julia REPL.
OpenMDAOCore.jl is a fairly small package without any Python dependencies, but OpenMDAO.jl depends on `omjlcomps` and `openmdao` itself.
OpenMDAO.jl's Python dependencies are managed by [`CondaPkg`](https://github.com/cjdoris/CondaPkg.jl), and should be automatically installed into a separate Conda environment specific to your current Julia environment.

## Acknowledgements
* An early version of OpenMDAO.jl was written by Daniel Ingraham, Justin Gray, and Andrew Ning while visiting Prof. Ning at Brigham Young University.
* OpenMDAO.jl depends heavily on [PythonCall and related packages](https://github.com/cjdoris/PythonCall.jl), developed by Christopher Rowley.
