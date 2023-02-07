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
