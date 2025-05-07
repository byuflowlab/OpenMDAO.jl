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

## OpenMDAOCore.jl must be explicitly installed in Julia projects/environments that use OpenMDAO.jl
OpenMDAO.jl does something strange in its `__init__`: it calls `Pkg.add("OpenMDAOCore")`, which has the effect of explicitly installing OpenMDAOCore.jl in the current Julia project/environment.
Why?
Well, let's remove that call, install OpenMDAO.jl in a fresh environment, and see what happens.

```
shell> mkdir MyNewEnvironment

shell> cd MyNewEnvironment/
/home/dingraha/Desktop/MyNewEnvironment

(blah) pkg> activate .
  Activating new project at `~/Desktop/MyNewEnvironment`

(MyNewEnvironment) pkg> status
Status `~/Desktop/MyNewEnvironment/Project.toml` (empty project)

(MyNewEnvironment) pkg> dev --local OpenMDAO
     Cloning git-repo `https://github.com/byuflowlab/OpenMDAO.jl.git`
   Resolving package versions...
    Updating `~/Desktop/MyNewEnvironment/Project.toml`
  [2d3f9b48] + OpenMDAO v0.4.1 `dev/OpenMDAO/julia/OpenMDAO.jl`
    Updating `~/Desktop/MyNewEnvironment/Manifest.toml`
  [992eb4ea] + CondaPkg v0.2.28
  [9a962f9c] + DataAPI v1.16.0
  [e2d170a0] + DataValueInterfaces v1.0.0
  [82899510] + IteratorInterfaceExtensions v1.0.0
  [692b3bcd] + JLLWrappers v1.7.0
  [0f8b85d8] + JSON3 v1.14.2
  [1914dd2f] + MacroTools v0.5.16
  [0b3b1443] + MicroMamba v0.1.14
  [2d3f9b48] + OpenMDAO v0.4.1 `dev/OpenMDAO/julia/OpenMDAO.jl`
  [24d19c10] + OpenMDAOCore v0.3.1
  [bac558e1] + OrderedCollections v1.8.0
  [69de0a69] + Parsers v2.8.3
  [fa939f87] + Pidfile v1.3.0
⌅ [aea7be01] + PrecompileTools v1.2.1
  [21216c6a] + Preferences v1.4.3
  [6099a3de] + PythonCall v0.9.24
  [ae029012] + Requires v1.3.1
  [6c6a2e73] + Scratch v1.2.1
  [856f2bd8] + StructTypes v1.11.0
  [3783bdb8] + TableTraits v1.0.1
  [bd369af6] + Tables v1.12.0
  [e17b2a0c] + UnsafePointers v1.0.0
  [f8abcde7] + micromamba_jll v1.5.8+0
  [4d7b5844] + pixi_jll v0.41.3+0
  [0dad84c5] + ArgTools v1.1.2
  [56f22d72] + Artifacts v1.11.0
  [2a0f44e3] + Base64 v1.11.0
  [ade2ca70] + Dates v1.11.0
  [f43a241f] + Downloads v1.6.0
  [7b1f6079] + FileWatching v1.11.0
  [b77e0a4c] + InteractiveUtils v1.11.0
  [4af54fe1] + LazyArtifacts v1.11.0
  [b27032c2] + LibCURL v0.6.4
  [76f85450] + LibGit2 v1.11.0
  [8f399da3] + Libdl v1.11.0
  [56ddb016] + Logging v1.11.0
  [d6f4376e] + Markdown v1.11.0
  [a63ad114] + Mmap v1.11.0
  [ca575930] + NetworkOptions v1.2.0
  [44cfe95a] + Pkg v1.11.0
  [de0858da] + Printf v1.11.0
  [9a3f8284] + Random v1.11.0
  [ea8e919c] + SHA v0.7.0
  [9e88b42a] + Serialization v1.11.0
  [fa267f1f] + TOML v1.0.3
  [a4e569a6] + Tar v1.10.0
  [8dfed614] + Test v1.11.0
  [cf7118a7] + UUIDs v1.11.0
  [4ec0a83e] + Unicode v1.11.0
  [deac9b47] + LibCURL_jll v8.6.0+0
  [e37daf67] + LibGit2_jll v1.7.2+0
  [29816b5a] + LibSSH2_jll v1.11.0+1
  [c8ffd9c3] + MbedTLS_jll v2.28.6+0
  [14a3606d] + MozillaCACerts_jll v2023.12.12
  [83775a58] + Zlib_jll v1.2.13+1
  [8e850ede] + nghttp2_jll v1.59.0+0
  [3f19e933] + p7zip_jll v17.4.0+2
        Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `s
tatus --outdated -m`

shell> nvr ./dev/OpenMDAO/julia/OpenMDAO.jl/src/OpenMDAO.jl # edit OpenMDAO.jl, removing `Pkg.add("OpenMDAOCore")`

(MyNewEnvironment) pkg> status
Status `~/Desktop/MyNewEnvironment/Project.toml`
  [2d3f9b48] OpenMDAO v0.4.1 `dev/OpenMDAO/julia/OpenMDAO.jl`

julia> using OpenMDAO
Precompiling OpenMDAO...
Info Given OpenMDAO was explicitly requested, output will be shown live 
    CondaPkg Found dependencies: /home/dingraha/.julia/packages/PythonCall/WMWY0/CondaPkg.toml
    CondaPkg Found dependencies: /home/dingraha/Desktop/MyNewEnvironment/dev/OpenMDAO/julia/OpenMDAO.jl/CondaPkg.toml
    CondaPkg Resolving changes
             + juliapkg (pip)
             + libstdcxx-ng
             + omjlcomps (pip)
             + openmdao
             + python
             + uv
    CondaPkg Initialising pixi
             │ /home/dingraha/.julia/artifacts/cefba4912c2b400756d043a2563ef77a0088866b/bin/pixi
             │ init
             │ --format pixi
             └ /home/dingraha/Desktop/MyNewEnvironment/.CondaPkg
✔ Created /home/dingraha/Desktop/MyNewEnvironment/.CondaPkg/pixi.toml
    CondaPkg Wrote /home/dingraha/Desktop/MyNewEnvironment/.CondaPkg/pixi.toml
             │ [dependencies]
             │ uv = ">=0.4"
             │ libstdcxx-ng = ">=3.4,<13.0"
             │ openmdao = ">=3.26.0,<4"
             │ 
             │     [dependencies.python]
             │     channel = "conda-forge"
             │     build = "*cpython*"
             │     version = ">=3.8,<4"
             │ 
             │ [project]
             │ name = ".CondaPkg"
             │ platforms = ["linux-64"]
             │ channels = ["conda-forge"]
             │ channel-priority = "strict"
             │ description = "automatically generated by CondaPkg.jl"
             │ 
             │ [pypi-dependencies]
             │ juliapkg = "~=0.1.10"
             └ omjlcomps = "~=0.2.0"
    CondaPkg Installing packages
             │ /home/dingraha/.julia/artifacts/cefba4912c2b400756d043a2563ef77a0088866b/bin/pixi
             │ install
             └ --manifest-path /home/dingraha/Desktop/MyNewEnvironment/.CondaPkg/pixi.toml
✔ The default environment has been installed.
  4 dependencies successfully precompiled in 24 seconds. 48 already precompiled.
  1 dependency had output during precompilation:
┌ OpenMDAO
│  [Output was shown above]
└  
ERROR: InitError: Python: Julia: ArgumentError: Package OpenMDAOCore not found in current path.
- Run `import Pkg; Pkg.add("OpenMDAOCore")` to install the OpenMDAOCore package.
Stacktrace:
  [1] macro expansion
    @ ./loading.jl:2296 [inlined]
  [2] macro expansion
    @ ./lock.jl:273 [inlined]
  [3] __require(into::Module, mod::Symbol)
    @ Base ./loading.jl:2271
  [4] #invoke_in_world#3
    @ ./essentials.jl:1089 [inlined]
  [5] invoke_in_world
    @ ./essentials.jl:1086 [inlined]
  [6] require(into::Module, mod::Symbol)
    @ Base ./loading.jl:2260
  [7] eval
    @ ./boot.jl:430 [inlined]
  [8] eval
    @ ./Base.jl:130 [inlined]
  [9] pyjlmodule_seval(self::Module, expr::PythonCall.Core.Py)
    @ PythonCall.JlWrap ~/.julia/packages/PythonCall/WMWY0/src/JlWrap/module.jl:13
 [10] _pyjl_callmethod(f::Any, self_::Ptr{PythonCall.C.PyObject}, args_::Ptr{PythonCall.C.PyObject}, nargs::Int64)
    @ PythonCall.JlWrap ~/.julia/packages/PythonCall/WMWY0/src/JlWrap/base.jl:67
 [11] _pyjl_callmethod(o::Ptr{PythonCall.C.PyObject}, args::Ptr{PythonCall.C.PyObject})
    @ PythonCall.JlWrap.Cjl ~/.julia/packages/PythonCall/WMWY0/src/JlWrap/C.jl:63
 [12] PyImport_Import
    @ ~/.julia/packages/PythonCall/WMWY0/src/C/pointers.jl:303 [inlined]
 [13] macro expansion
    @ ~/.julia/packages/PythonCall/WMWY0/src/Core/Py.jl:132 [inlined]
 [14] pyimport(m::String)
    @ PythonCall.Core ~/.julia/packages/PythonCall/WMWY0/src/Core/builtins.jl:1561
 [15] __init__()
    @ OpenMDAO ~/Desktop/MyNewEnvironment/dev/OpenMDAO/julia/OpenMDAO.jl/src/OpenMDAO.jl:16
 [16] run_module_init(mod::Module, i::Int64)
    @ Base ./loading.jl:1378
 [17] register_restored_modules(sv::Core.SimpleVector, pkg::Base.PkgId, path::String)
    @ Base ./loading.jl:1366
 [18] _include_from_serialized(pkg::Base.PkgId, path::String, ocachepath::String, depmods::Vector{Any}, ignore_native::Nothing; register::Boo
l)
    @ Base ./loading.jl:1254
 [19] _include_from_serialized (repeats 2 times)
    @ ./loading.jl:1210 [inlined]
 [20] _require_search_from_serialized(pkg::Base.PkgId, sourcepath::String, build_id::UInt128, stalecheck::Bool; reasons::Dict{…}, DEPOT_PATH:
:Vector{…})
    @ Base ./loading.jl:2057
 [21] _require(pkg::Base.PkgId, env::String)
    @ Base ./loading.jl:2527
 [22] __require_prelocked(uuidkey::Base.PkgId, env::String)
    @ Base ./loading.jl:2388
 [23] #invoke_in_world#3
    @ ./essentials.jl:1089 [inlined]
 [24] invoke_in_world
    @ ./essentials.jl:1086 [inlined]
 [25] _require_prelocked(uuidkey::Base.PkgId, env::String)
    @ Base ./loading.jl:2375
 [26] macro expansion
    @ ./loading.jl:2314 [inlined]
 [27] macro expansion
    @ ./lock.jl:273 [inlined]
 [28] __require(into::Module, mod::Symbol)
    @ Base ./loading.jl:2271
 [29] #invoke_in_world#3
    @ ./essentials.jl:1089 [inlined]
 [30] invoke_in_world
    @ ./essentials.jl:1086 [inlined]
 [31] require(into::Module, mod::Symbol)
    @ Base ./loading.jl:2260
 [32] eval
    @ ./boot.jl:430 [inlined]
 [33] eval_user_input(ast::Any, backend::REPL.REPLBackend, mod::Module)
    @ REPL ~/local/julia/1.11.5/share/julia/stdlib/v1.11/REPL/src/REPL.jl:261
 [34] repl_backend_loop(backend::REPL.REPLBackend, get_module::Function)
    @ REPL ~/local/julia/1.11.5/share/julia/stdlib/v1.11/REPL/src/REPL.jl:368
 [35] start_repl_backend(backend::REPL.REPLBackend, consumer::Any; get_module::Function)
    @ REPL ~/local/julia/1.11.5/share/julia/stdlib/v1.11/REPL/src/REPL.jl:343
 [36] run_repl(repl::REPL.AbstractREPL, consumer::Any; backend_on_current_task::Bool, backend::Any)
    @ REPL ~/local/julia/1.11.5/share/julia/stdlib/v1.11/REPL/src/REPL.jl:500
 [37] run_repl(repl::REPL.AbstractREPL, consumer::Any)
    @ REPL ~/local/julia/1.11.5/share/julia/stdlib/v1.11/REPL/src/REPL.jl:486
 [38] (::Base.var"#1150#1152"{Bool, Symbol, Bool})(REPL::Module)
    @ Base ./client.jl:446
 [39] #invokelatest#2
    @ ./essentials.jl:1055 [inlined]
 [40] invokelatest
    @ ./essentials.jl:1052 [inlined]
 [41] run_main_repl(interactive::Bool, quiet::Bool, banner::Symbol, history_file::Bool, color_set::Bool)
    @ Base ./client.jl:430
 [42] repl_main
    @ ./client.jl:567 [inlined]
 [43] _start()
    @ Base ./client.jl:541
Python stacktrace:
 [1] seval
   @ ~/.julia/packages/PythonCall/WMWY0/src/JlWrap/module.jl:27
 [2] <module>
   @ ~/Desktop/MyNewEnvironment/.CondaPkg/.pixi/envs/default/lib/python3.12/site-packages/omjlcomps/__init__.py:9
Stacktrace:
  [1] pythrow()
    @ PythonCall.Core ~/.julia/packages/PythonCall/WMWY0/src/Core/err.jl:92
  [2] errcheck
    @ ~/.julia/packages/PythonCall/WMWY0/src/Core/err.jl:10 [inlined]
  [3] pyimport(m::String)
    @ PythonCall.Core ~/.julia/packages/PythonCall/WMWY0/src/Core/builtins.jl:1561
  [4] __init__()
    @ OpenMDAO ~/Desktop/MyNewEnvironment/dev/OpenMDAO/julia/OpenMDAO.jl/src/OpenMDAO.jl:16
  [5] run_module_init(mod::Module, i::Int64)
    @ Base ./loading.jl:1378
  [6] register_restored_modules(sv::Core.SimpleVector, pkg::Base.PkgId, path::String)
    @ Base ./loading.jl:1366
  [7] _include_from_serialized(pkg::Base.PkgId, path::String, ocachepath::String, depmods::Vector{Any}, ignore_native::Nothing; register::Boo
l)
    @ Base ./loading.jl:1254
  [8] _include_from_serialized (repeats 2 times)
    @ ./loading.jl:1210 [inlined]
  [9] _require_search_from_serialized(pkg::Base.PkgId, sourcepath::String, build_id::UInt128, stalecheck::Bool; reasons::Dict{…}, DEPOT_PATH:
:Vector{…})
    @ Base ./loading.jl:2057
 [10] _require(pkg::Base.PkgId, env::String)
    @ Base ./loading.jl:2527
 [11] __require_prelocked(uuidkey::Base.PkgId, env::String)
    @ Base ./loading.jl:2388
 [12] #invoke_in_world#3
    @ ./essentials.jl:1089 [inlined]
 [13] invoke_in_world
    @ ./essentials.jl:1086 [inlined]
 [14] _require_prelocked(uuidkey::Base.PkgId, env::String)
    @ Base ./loading.jl:2375
 [15] macro expansion
    @ ./loading.jl:2314 [inlined]
 [16] macro expansion
    @ ./lock.jl:273 [inlined]
 [17] __require(into::Module, mod::Symbol)
    @ Base ./loading.jl:2271
 [18] #invoke_in_world#3
    @ ./essentials.jl:1089 [inlined]
 [19] invoke_in_world
    @ ./essentials.jl:1086 [inlined]
 [20] require(into::Module, mod::Symbol)
    @ Base ./loading.jl:2260
during initialization of module OpenMDAO
Some type information was truncated. Use `show(err)` to see complete types.

julia> 
```

What happened there?
We see from the output when installing `OpenMDAO.jl` that `OpenMDAOCore.jl` was installed, and when doing `using OpenMDAO` the Python environment was set up nicely for us, and did install the "real" Python OpenMDAO and `omjlcomps` packages for us.
But then `omjlcomps` complains about not being able to find OpenMDAOCore, despite it being in the `MyNewEnvironment` `Manifest.toml`:

```
(MyNewEnvironment) pkg> status --manifest
Status `~/Desktop/MyNewEnvironment/Manifest.toml`
  [992eb4ea] CondaPkg v0.2.28
  [9a962f9c] DataAPI v1.16.0
  [e2d170a0] DataValueInterfaces v1.0.0
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.7.0
  [0f8b85d8] JSON3 v1.14.2
  [1914dd2f] MacroTools v0.5.16
  [0b3b1443] MicroMamba v0.1.14
  [2d3f9b48] OpenMDAO v0.4.1 `dev/OpenMDAO/julia/OpenMDAO.jl`
  [24d19c10] OpenMDAOCore v0.3.1
  [bac558e1] OrderedCollections v1.8.0
  [69de0a69] Parsers v2.8.3
  [fa939f87] Pidfile v1.3.0
⌅ [aea7be01] PrecompileTools v1.2.1
  [21216c6a] Preferences v1.4.3
  [6099a3de] PythonCall v0.9.24
  [ae029012] Requires v1.3.1
  [6c6a2e73] Scratch v1.2.1
  [856f2bd8] StructTypes v1.11.0
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.12.0
  [e17b2a0c] UnsafePointers v1.0.0
  [f8abcde7] micromamba_jll v1.5.8+0
  [4d7b5844] pixi_jll v0.41.3+0
  [0dad84c5] ArgTools v1.1.2
  [56f22d72] Artifacts v1.11.0
  [2a0f44e3] Base64 v1.11.0
  [ade2ca70] Dates v1.11.0
  [f43a241f] Downloads v1.6.0
  [7b1f6079] FileWatching v1.11.0
  [b77e0a4c] InteractiveUtils v1.11.0
  [4af54fe1] LazyArtifacts v1.11.0
  [b27032c2] LibCURL v0.6.4
  [76f85450] LibGit2 v1.11.0
  [8f399da3] Libdl v1.11.0
  [56ddb016] Logging v1.11.0
  [d6f4376e] Markdown v1.11.0
  [a63ad114] Mmap v1.11.0
  [ca575930] NetworkOptions v1.2.0
  [44cfe95a] Pkg v1.11.0
  [de0858da] Printf v1.11.0
  [9a3f8284] Random v1.11.0
  [ea8e919c] SHA v0.7.0
  [9e88b42a] Serialization v1.11.0
  [fa267f1f] TOML v1.0.3
  [a4e569a6] Tar v1.10.0
  [8dfed614] Test v1.11.0
  [cf7118a7] UUIDs v1.11.0
  [4ec0a83e] Unicode v1.11.0
  [deac9b47] LibCURL_jll v8.6.0+0
  [e37daf67] LibGit2_jll v1.7.2+0
  [29816b5a] LibSSH2_jll v1.11.0+1
  [c8ffd9c3] MbedTLS_jll v2.28.6+0
  [14a3606d] MozillaCACerts_jll v2023.12.12
  [83775a58] Zlib_jll v1.2.13+1
  [8e850ede] nghttp2_jll v1.59.0+0
  [3f19e933] p7zip_jll v17.4.0+2
Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --
outdated -m`

(MyNewEnvironment) pkg> 
```

But, now, let's explicitly install `OpenMDAOCore.jl`:

```
(MyNewEnvironment) pkg> add OpenMDAOCore
   Resolving package versions...
    Updating `~/Desktop/MyNewEnvironment/Project.toml`
  [24d19c10] + OpenMDAOCore v0.3.1
  No Changes to `~/Desktop/MyNewEnvironment/Manifest.toml`

julia> using OpenMDAO

julia> 
```

Now things seem to work!
What's going on?
I'm not entirely sure.
For some reason, `omjlcomps` isn't able to find the OpenMDAOCore.jl that is installed by OpenMDAO.jl "behind the scenes" when it's a plain dependency, but can when it's explicitly installed.
So, the `Pkg.add("OpenMDAOCore")` call does this for us automatically.
It will install the latest stable version of OpenMDAOCore.jl that's compatible with your environment.
If you require a specific version of OpenMDAOCore.jl (eg you want to `dev` it while working on it), you'll need to install that in your environment in the usual way with Julia's package manager.

## Finding `juliapkg.json` files in editable Python packages
[`juliapkg`](https://github.com/JuliaPy/pyjuliapkg) is the Python package that allows us to declare Julia dependencies in Python packages.
We do this by including `juliapkg.json` files in our Python packages that describe the Julia dependencies.
`juliapkg` will then look for these JSON files in, among other locations, "every directory and direct sub-directory in `sys.path`.
When installing Python packages in "editable" mode with `pip install --editable` or `pip install -e`, though, the package is generally **not** added to `sys.path`, and so `juliapkg` will not find `juliapkg.json` will not find the JSON file, not install the required Julia dependencies, and finally you'll expect to see errors where Julia complains that it can't find the Julia packages your code requires.
A fix for this has been included in [this PR](https://github.com/JuliaPy/pyjuliapkg/pull/52), but in the meantime, the solution appears to be to install the editable Python package with `pip install --config-settings editable_mode=compat -e <path to package>`.

## Distributing Julia files in a Python package
When creating a source distribution for a Python package, `setuptools` will only include [certain files](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html), and, perhaps not surprisingly, will ignore Julia files.
The solution is to add a `MANIFEST.in` to the top-level of the Python package that includes all the Julia files you need.
See the example package in `examples` for an... example of how to do this.
