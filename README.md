OpenMDAO.jl
===========

[![Tests](https://github.com/dingraha/OpenMDAO.jl/actions/workflows/CI.yaml/badge.svg?branch=pythoncall_juliacall)](https://github.com/dingraha/OpenMDAO.jl/actions/workflows/CI.yaml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://dingraha.github.io/OpenMDAO.jl/dev)

What?
-----
Use Julia with [OpenMDAO](https://openmdao.org/)! OpenMDAO.jl is a Julia package that allows a user to:

  * Write OpenMDAO `Component`s in Julia, and incorporate these components into a
    OpenMDAO optimization.
  * Create OpenMDAO optimizations (i.e., run scripts) in Julia, using OpenMDAO
    as a library.

How? (Installation Instructions)
--------------------------------
Prerequisites: 

  * Python: a reasonably recent version that OpenMDAO supports.
  * Julia: install instructions can be found
    [here](https://julialang.org/downloads/platform/). Make sure typing `julia`
    in your shell will get you to the Julia REPL. Should just need to make sure
    that the `julia` binary is in your `$PATH`.

Recommended:

  * Some way of managing your Python environment. I use
    [virtualenv](https://github.com/pypa/virtualenv), but Conda would probibaly
    be OK too.

Let's assume you're at your shell prompt, in a directory where you'd like your
new OpenMDAO.jl project to live. Step 1 is to create and activate a new Python
`virtualenv` for your project:

```
% virtualenv dev/venv
created virtual environment CPython3.7.9.final.0-64 in 2340ms
  creator CPython3Posix(dest=/home/dingraha/projects/openmdao.jl_install_test/dev/venv, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/dingraha/.local/share/virtualenv)
    added seed packages: pip==20.3.3, setuptools==51.0.0, wheel==0.36.2
  activators BashActivator,CShellActivator,FishActivator,PowerShellActivator,PythonActivator,XonshActivator
% echo 'export PYCALL_JL_RUNTIME_PYTHON="dev/venv/bin/python"' >> dev/venv/bin/activate
% . ./dev/venv/bin/activate
```

The line with the [`PYCALL_JL_RUNTIME_PYTHON`
variable](https://github.com/JuliaPy/PyCall.jl#python-virtual-environments) is
used to tell the Julia PyCall package to use the new virtual environment's
Python, and not any other one found in the system's `PATH`.

Let's also tell Julia what project we're going to use.
```
(venv) % export JULIA_PROJECT=$PWD
(venv) % echo "export JULIA_PROJECT=$JULIA_PROJECT" >> $VIRTUAL_ENV/bin/activate"
(venv) %
```

`OpenMDAO.jl` needs OpenMDAO itself, so install that in your new virtual
environment with `pip`. This might not be necessary if you use Conda to manage
your Python installation—OpenMDAO.jl uses the [`pyimport_conda`
function](https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules)
to access OpenMDAO, which should install OpenMDAO behind the scenes with Conda.

```
(venv) % pip install OpenMDAO
<lots of output removed>
(venv) %
```

And we also need the [`PyJulia` package](https://github.com/JuliaPy/pyjulia),
which makes it so we can call Julia code from Python.
```
(venv) % pip install julia
Collecting julia
  Using cached julia-0.5.6-py2.py3-none-any.whl (67 kB)
Installing collected packages: julia
Successfully installed julia-0.5.6
(venv) %
```

Fire up Julia and install OpenMDAO. After you type `julia` and hit Enter, type
`]` to get to the `Pkg` prompt, which is Julia's way of managing packages and
"Projects" (Julia's equivalent of Python virtual environments). (Read more about
Pkg [here](https://docs.julialang.org/en/v1/stdlib/Pkg/)—it is really great.)
The `dev --local` command downloads the OpenMDAO.jl repository from GitHub and
installs it in the `dev` directory inside the current active Julia Project
(currently set to `$JULIA_PROJECT` above).

```
(venv) % julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.5.3 (2020-11-09)
 _/ |\__'_|_|_|\__'_|  |
|__/                   |

(openmdao.jl_install_test) pkg> dev --local https://github.com/byuflowlab/OpenMDAO.jl
    Cloning git-repo `https://github.com/byuflowlab/OpenMDAO.jl`
  Resolving package versions...
Updating `~/projects/openmdao.jl_install_test/Project.toml`
  [2d3f9b48] + OpenMDAO v0.1.0 `dev/OpenMDAO`
Updating `~/projects/openmdao.jl_install_test/Manifest.toml`
  [8f4d0f93] + Conda v1.5.0
  [682c06a0] + JSON v0.21.1
  [1914dd2f] + MacroTools v0.5.6
  [2d3f9b48] + OpenMDAO v0.1.0 `dev/OpenMDAO`
  [69de0a69] + Parsers v1.0.15
  [438e738f] + PyCall v1.92.2
  [81def892] + VersionParsing v1.2.0
  [2a0f44e3] + Base64
  [ade2ca70] + Dates
  [8ba89e20] + Distributed
  [b77e0a4c] + InteractiveUtils
  [8f399da3] + Libdl
  [37e2e46d] + LinearAlgebra
  [56ddb016] + Logging
  [d6f4376e] + Markdown
  [a63ad114] + Mmap
  [de0858da] + Printf
  [9a3f8284] + Random
  [9e88b42a] + Serialization
  [6462fe0b] + Sockets
  [8dfed614] + Test
  [4ec0a83e] + Unicode

(openmdao.jl_install_test) pkg>
```

Now test the OpenMDAO.jl package from the Julia Pkg prompt to make sure
everything is OK.
```
(openmdao.jl_install_test) pkg> test OpenMDAO
    Testing OpenMDAO
<lots of output removed>
    Testing OpenMDAO tests passed

(openmdao.jl_install_test) pkg>
```

That's it! Check out the OpenMDAO.jl examples, which should be in
`dev/OpenMDAO/examples`.
