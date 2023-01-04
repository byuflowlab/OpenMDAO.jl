# OpenMDAO.jl Documentation

## What?
Use Julia with [OpenMDAO](https://openmdao.org/)!
OpenMDAO.jl is a Julia package that allows a user to:

  * Write OpenMDAO `Component`s in Julia, and incorporate these components into a OpenMDAO model.
  * Create and run optimizations in Julia, using OpenMDAO as a library.

OpenMDAO.jl consists of three pieces of software:

  * OpenMDAOCore.jl: A small, pure-Julia package that allows users to define Julia code that will eventually be used in an OpenMDAO `Problem`. OpenMDAOCore.jl defines two Julia abstract types (`AbstractExplicitComponent` and `AbstractImplicitComponent`) and methods that mimic OpenMDAO's ExplicitComponent and ImplicitComponent classes.
  * omjlcomps: A Python package (actually, a [OpenMDAO Plugin](https://openmdao.org/newdocs/versions/latest/features/experimental/plugins.html)) that defines two classes, `JuliaExplicitComp` and `JuliaImplicitComp`, which inherit from OpenMDAO's `ExplicitComponent` and `ImplicitComponent`, respectively.
    These components takes instances of concrete subtypes of `OpenMDAOCore.ExplicitComponent` and `OpenMDAOCore.ImplicitComponent` and turn them into instances of `JuliaExplicitComp` and `JuliaImplicitComp`.
    Like any other OpenMDAO `ExplicitComponent` or `ImplicitComponent` objects, `JuliaExplicitComp` and `JuliaImplicitComp` instances can be used in an OpenMDAO model, but call Julia code in their methods (`compute`, `apply_nonlinear`, etc.).
  * OpenMDAO.jl: A Julia package that has the openmdao and omjlcomps Python packages as dependencies.
    Users can install `OpenMDAO.jl` and have the full power of the OpenMDAO framework at their disposal in Julia.

## How?
There are two approaches to getting OpenMDAO working with Julia: the [Python-centric approach](@ref python_centric) and the Julia-centric approach.
If you like Python and just want to have a bit of Julia buried in your OpenMDAO `model`, then you'll probably prefer the [Python-centric approach].
If you're a huge fan of Julia and would like to pretend that OpenMDAO is a Julia library, you'll want the Julia-centric approach.
