```@meta
CurrentModule = OpenMDAODocs
```
# Developer Docs

## How to Release a New Version
For either OpenMDAOCore.jl or OpenMDAO.jl, registering a new version should be as simple as commenting

> @JuliaRegistrator register subdir=julia/OpenMDAOCore.jl

or

> @JuliaRegistrator register subdir=julia/OpenMDAO.jl

on a new issue, like [here](https://github.com/byuflowlab/OpenMDAO.jl/issues/22).
Be sure to adjust the `version` field in the appropriate `Project.toml` before you do that.
And after the new version is registered, don't forget to tag it as suggested by the JuliaRegistrator bot.
For example, OpenMDAOCore.jl version 0.3.1 was tagged like this:

```
$ git tag OpenMDAOCore.jl-v0.3.1 ea03a4e1be02a989021e5b466fc1fe51534e6fdb
$ git push upstream OpenMDAOCore.jl-v0.3.1
```

where `upstream` is the remote corresponding to `byuflowlab/OpenMDAO.jl.git`:

```
$ git remote -v
origin  git@github.com:dingraha/OpenMDAO.jl.git (fetch)
origin  git@github.com:dingraha/OpenMDAO.jl.git (push)
upstream        git@github.com:byuflowlab/OpenMDAO.jl.git (fetch)
upstream        git@github.com:byuflowlab/OpenMDAO.jl.git (push)
$
```

For `omjlcomps`, registration is done by manually running the "Register to PyPI" workflow from the GitHub Actions tab (basically copied from [PythonCall.jl](https://github.com/cjdoris/PythonCall.jl)).
Be sure to adjust the version in the `python/setup.py` file before registering a new version.
After clicking on the Actions tab on https://github.com/byuflowlab/OpenMDAO.jl, you'll see "Register to PyPI" listed under "All workflows" on the left-hand side.
Click on that, then click on the "Run workflow" dropdown button on the right-hand side of the screen.
Run it from the master branch, then wait for the workflow to finish.
After that, you should see the new version of `omjlcomps` on PyPI: https://pypi.org/project/omjlcomps/.
Once that's done, tag the commit that contains the new version of `omjlcomps` and push that to `byuflowlab/OpenMDAO.jl.git`.
For example, `omjlcomps` version 0.2.3 was tagged with

```
$ git tag omjlcomps-v0.2.3 d7830552dc3d54fe18b89dc91b36219739d13a62
$ git push upstream omjlcomps-v0.2.3
```

where `upstream` is the remote corresponding to `byuflowlab/OpenMDAO.jl.git`:

```
$ git remote -v
origin  git@github.com:dingraha/OpenMDAO.jl.git (fetch)
origin  git@github.com:dingraha/OpenMDAO.jl.git (push)
upstream        git@github.com:byuflowlab/OpenMDAO.jl.git (fetch)
upstream        git@github.com:byuflowlab/OpenMDAO.jl.git (push)
$
```

## How to Release a New Version (Old, LocalRegistry Way)
> **Note**
> This section of the docs describes how I released new versions of `OpenMDAO{,Core}.jl` and `omjlcomps` before getting stuff registered in the Julia General registry.
> They are outdated and unnecessary, but I'm keeping them for now in case someone wants to do something similar down the road.

It's a bit tricky to release a new version of OpenMDAO.jl, since this repository contains 3 separate software packages: the Julia packages OpenMDAOCore.jl and OpenMDAO.jl, and the Python package omjlcomps.
Here's how to do it.

### Step 1: Hack on OpenMDAO.jl
The first step of releasing a new version is obviously hacking on the code, which is no different than working on any other piece of software on GitHub.
You'll need to fork the `byuflowlab/OpenMDAO.jl` GitHub repository, then clone it to your local machine with

```
$ git clone git@github.com:<your_user_name>/OpenMDAO.jl.git
```

Then you can start making changes, hopefully in a new local feature branch you create.

### Step 2: Submit a PR to `byuflowlab/OpenMDAO.jl.git` and merge
After you're happy with your changes (they have tests, right? And pass those tests?), push your local changes to your GitHub fork, and then submit a PR to `byuflowlab/OpenMDAO.jl.git`.
Be sure to bump the version number in each sub-package appropriately (if necessary: no need to modify all three version numbers just because a change was made to only one or two of the packages).
The OpenMDAOCore.jl and OpenMDAO.jl version numbers should follow semantic versioning (check out the [Julia Pkg docs on compatibility](https://pkgdocs.julialang.org/v1/compatibility/)).
I try to follow semantic versioning with the `omjlcomps` Python package, even though semantic versioning doesn't appear to be as pervasive in the Python ecosystem as it is in Julia land.

After everyone is satisfied with the PR, an Administrator of the `byuflowab/OpenMDAO.jl` repository will merge the package.

### Step 3: Tag the new version(s)
Every time the version of any of the three OpenMDAO.jl sub-packages is bumped, we should tag a new version.
The way to do this is the following:

  * First, check out the upstream `master` branch, i.e. if the remote branch pointing to `byuflowlab/OpenMDAO.jl.git` is `upstream`, then do `git checkout upstream/master`.
    That should put the local repository in a "detached HEAD" state, like this:

    ```
    $ git checkout upstream/master
    Note: switching to 'upstream/master'.

    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by switching back to a branch.

    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -c with the switch command. Example:

      git switch -c <new-branch-name>

    Or undo this operation with:

      git switch -

    Turn off this advice by setting config variable advice.detachedHead to false

    HEAD is now at a9e8f78 Merge pull request #16 from dingraha/licence
    $
    ```

  * Make sure you're on the commit corresponding to the merged pull request you want to release/tag, using, for example, `git log -n1`:

    ```
    $ git log -n 1
    commit a9e8f7849844640f77a2eadd3683476be92ba8fb (HEAD, upstream/master, origin/master, origin/HEAD, master, how_to_release)
    Merge: 16f0355 054b78c
    Author: Daniel Ingraham <d.j.ingraham@gmail.com>
    Date:   Thu May 25 09:02:07 2023 -0400

        Merge pull request #16 from dingraha/licence
        
        Add LICENSE
    $ 
    ```

  * Create the tag(s) using `git tag`.
    The tag or tags should follow the format `package_name-vX.Y.Z`.
    For example, here are the tags at the time of writing:

    ```
    $ git tag
    OpenMDAO.jl-v0.3.0
    OpenMDAO.jl-v0.3.1
    OpenMDAO.jl-v0.3.2
    OpenMDAO.jl-v0.4.0
    OpenMDAOCore.jl-v0.2.10
    OpenMDAOCore.jl-v0.2.8
    OpenMDAOCore.jl-v0.2.9
    OpenMDAOCore.jl-v0.3.0
    omjlcomps-v0.1.7
    omjlcomps-v0.1.8
    omjlcomps-v0.1.9
    omjlcomps-v0.2.0
    v0.2.0
    v0.2.1
    $ 
    ```

    (The `v0.2.0` and `v0.2.1` tags are from very early versions of OpenMDAO.jl and don't follow the recomended tag format.)

  * With the tags created, all that's left is to push them to `upstream`:

    ```
    $ git push --tags upstream
    ```

### Step 4: Release the new version(s) to the appropriate registries
For either of the two Julia packages (OpenMDAO.jl and OpenMDAOCore.jl), we need to register a new version with `DanielIngrahamRegistry` at `dingraha/DanielIngrahamRegistry` on GitHub using `LocalRegistry`.
Again, this is unfortunately something only I (Daniel) can do.
All that needs to be done is, from the Julia REPL:

  * Make sure you've `dev`ed the package you want to release, and are on the appropriate commit.
  * Do `LocalRegistry.register("OpenMDAOCore")` and/or `LocalRegistry.register("OpenMDAO")`, as appropriate.

If a new version of `omjlcomps` was tagged, then we need to put it on PyPI.
The way I do this currently is using `twine`:

```
$ # In the OpenMDAO.jl/python directory.
$ python setup.py sdist
running sdist
running egg_info
writing omjlcomps.egg-info/PKG-INFO
writing dependency_links to omjlcomps.egg-info/dependency_links.txt
writing entry points to omjlcomps.egg-info/entry_points.txt
writing requirements to omjlcomps.egg-info/requires.txt
writing top-level names to omjlcomps.egg-info/top_level.txt
reading manifest file 'omjlcomps.egg-info/SOURCES.txt'
reading manifest template 'MANIFEST.in'
writing manifest file 'omjlcomps.egg-info/SOURCES.txt'
running check
creating omjlcomps-0.1.9
creating omjlcomps-0.1.9/omjlcomps
creating omjlcomps-0.1.9/omjlcomps.egg-info
creating omjlcomps-0.1.9/omjlcomps/test
copying files to omjlcomps-0.1.9...
copying MANIFEST.in -> omjlcomps-0.1.9
copying README.md -> omjlcomps-0.1.9
copying setup.py -> omjlcomps-0.1.9
copying omjlcomps/__init__.py -> omjlcomps-0.1.9/omjlcomps
copying omjlcomps/juliapkg.json -> omjlcomps-0.1.9/omjlcomps
copying omjlcomps.egg-info/PKG-INFO -> omjlcomps-0.1.9/omjlcomps.egg-info
copying omjlcomps.egg-info/SOURCES.txt -> omjlcomps-0.1.9/omjlcomps.egg-info
copying omjlcomps.egg-info/dependency_links.txt -> omjlcomps-0.1.9/omjlcomps.egg-info
copying omjlcomps.egg-info/entry_points.txt -> omjlcomps-0.1.9/omjlcomps.egg-info
copying omjlcomps.egg-info/requires.txt -> omjlcomps-0.1.9/omjlcomps.egg-info
copying omjlcomps.egg-info/top_level.txt -> omjlcomps-0.1.9/omjlcomps.egg-info
copying omjlcomps/test/__init__.py -> omjlcomps-0.1.9/omjlcomps/test
copying omjlcomps/test/test_ecomp.jl -> omjlcomps-0.1.9/omjlcomps/test
copying omjlcomps/test/test_icomp.jl -> omjlcomps-0.1.9/omjlcomps/test
copying omjlcomps/test/test_julia_explicit_comp.py -> omjlcomps-0.1.9/omjlcomps/test
copying omjlcomps/test/test_julia_implicit_comp.py -> omjlcomps-0.1.9/omjlcomps/test
Writing omjlcomps-0.1.9/setup.cfg
Creating tar archive
removing 'omjlcomps-0.1.9' (and everything under it)
(venv) dingraha@GRLRL2021060743 ~/p/p/d/O/python (master %)> ls
build/  dist/  MANIFEST.in  omjlcomps/  omjlcomps.egg-info/  README.md  setup.py
(venv) dingraha@GRLRL2021060743 ~/p/p/d/O/python (master %)> twine upload dist/omjlcomps-0.1.9.tar.gz 
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: dingraha
Enter your password: 
Uploading omjlcomps-0.1.9.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.1/12.1 kB • 00:00 • 8.5 MB/s

View at:
https://pypi.org/project/omjlcomps/0.1.9/
$ 
```

That unfortunately is something only I can do, since it requires my username and password for PyPI.
Also it uses plain password authentication, which I think isn't supported anymore, or won't be for long.

