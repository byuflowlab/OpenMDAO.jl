using SafeTestsets: @safetestset

@safetestset "Explicit Components" begin

    @safetestset "simple" begin
        using OpenMDAO: om, make_component
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test

        struct ECompSimple <: OpenMDAOCore.AbstractExplicitComp end

        function OpenMDAOCore.setup(self::ECompSimple)
            input_data = [OpenMDAOCore.VarData("x")]
            output_data = [OpenMDAOCore.VarData("y")]
            partials_data = [OpenMDAOCore.PartialsData("y", "x")]

            return input_data, output_data, partials_data
        end

        function OpenMDAOCore.compute!(self::ECompSimple, inputs, outputs)
            outputs["y"][1] = 2*inputs["x"][1]^2 + 1
            return nothing
        end

        function OpenMDAOCore.compute_partials!(self::ECompSimple, inputs, partials)
            partials["y", "x"][1] = 4*inputs["x"][1]
            return nothing
        end

        p = om.Problem()
        ecomp = ECompSimple()
        comp = make_component(ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=true)
        p.set_val("x", 3.0)
        p.run_model()

        # Check that the outputs are what we expect.
        expected = 2 .* PyArray(p.get_val("x")).^2 .+ 1
        actual = PyArray(p.get_val("y"))
        @test actual ≈ expected

        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that the partials the user provided are what we expect.
        ecomp_partials = pyconvert(Dict, cpd["ecomp"])
        actual = pyconvert(Dict, ecomp_partials["y", "x"])["J_fwd"]
        expected = 4 .* PyArray(p.get_val("x"))
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end

    @safetestset "with option" begin
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test

        struct ECompWithOption <: OpenMDAOCore.AbstractExplicitComp
            a::Float64
        end

        function OpenMDAOCore.setup(self::ECompWithOption)
            input_data = [OpenMDAOCore.VarData("x")]
            output_data = [OpenMDAOCore.VarData("y")]
            partials_data = [OpenMDAOCore.PartialsData("y", "x")]

            return input_data, output_data, partials_data
        end

        function OpenMDAOCore.compute!(self::ECompWithOption, inputs, outputs)
            outputs["y"][1] = 2*self.a*inputs["x"][1]^2 + 1
            return nothing
        end

        function OpenMDAOCore.compute_partials!(self::ECompWithOption, inputs, partials)
            partials["y", "x"][1] = 4*self.a*inputs["x"][1]
            return nothing
        end

        p = om.Problem()
        a = 0.5
        ecomp = ECompWithOption(a)
        comp = make_component(ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=true)
        p.set_val("x", 3.0)
        p.run_model()

        # Check that the outputs are what we expect.
        expected = 2 .* a.*PyArray(p.get_val("x")).^2 .+ 1
        actual = PyArray(p.get_val("y"))
        @test actual ≈ expected

        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that the partials the user provided are what we expect.
        ecomp_partials = pyconvert(Dict, cpd["ecomp"])
        actual = pyconvert(Dict, ecomp_partials["y", "x"])["J_fwd"]
        expected = 4 .* a.*PyArray(p.get_val("x"))
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end

    @safetestset "matrix-free" begin
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test

        struct ECompMatrixFree <: OpenMDAOCore.AbstractExplicitComp
            nrows::Int
            ncols::Int
        end

        function OpenMDAOCore.setup(self::ECompMatrixFree)
            nrows = self.nrows
            ncols = self.ncols
            input_data = [OpenMDAOCore.VarData("x1"; shape=(nrows, ncols)), OpenMDAOCore.VarData("x2"; shape=(self.nrows, self.ncols))]
            output_data = [OpenMDAOCore.VarData("y1"; shape=(nrows, ncols)), OpenMDAOCore.VarData("y2"; shape=(self.nrows, self.ncols))]
            partials_data = [OpenMDAOCore.PartialsData("*", "*")]  # I think this should work.

            return input_data, output_data, partials_data
        end

        function OpenMDAOCore.compute!(self::ECompMatrixFree, inputs, outputs)
            x1, x2 = inputs["x1"], inputs["x2"]
            y1, y2 = outputs["y1"], outputs["y2"]
            @. y1 = 2*x1 + 3*x2^2
            @. y2 = 4*x1^3 + 5*x2^4
            return nothing
        end

        function OpenMDAOCore.compute_jacvec_product!(self::ECompMatrixFree, inputs, d_inputs, d_outputs, mode)
            x1, x2 = inputs["x1"], inputs["x2"]
            x1dot = get(d_inputs, "x1", nothing)
            x2dot = get(d_inputs, "x2", nothing)
            y1dot = get(d_outputs, "y1", nothing)
            y2dot = get(d_outputs, "y2", nothing)
            if mode == "fwd"
                # For forward mode, we are tracking the derivatives of everything with
                # respect to upstream inputs, and our goal is to calculate the
                # derivatives of this components outputs wrt the upstream inputs given
                # the derivatives of inputs wrt the upstream inputs.
                if y1dot !== nothing
                    fill!(y1dot, 0)
                    if x1dot !== nothing
                        @. y1dot += 2*x1dot
                    end
                    if x2dot !== nothing
                        @. y1dot += 6*x2*x2dot
                    end
                end
                if y2dot !== nothing
                    fill!(y2dot, 0)
                    if x1dot !== nothing
                        @. y2dot += 12*x1^2*x1dot
                    end
                    if x2dot !== nothing
                        @. y2dot += 20*x2^3*x2dot
                    end
                end
            elseif mode == "rev"
                # For reverse mode, we are tracking the derivatives of everything with
                # respect to a downstream output, and our goal is to calculate the
                # derivatives of the downstream output wrt each input given the
                # derivatives of the downstream output wrt each output.
                #
                # So, let's say I have a function f(y1, y2).
                # I start with fdot = df/df = 1.
                # Then I say that y1dot = df/dy1 = fdot*df/dy1
                # and y2dot = df/dy2 = fdot*df/dy2
                # Hmm...
                # f(y1(x1,x2), y2(x1, x2)) = df/dy1*(dy1/dx1 + dy1/dx2) + df/dy2*(dy2/dx1 + dy2/dx2)
                if x1dot !== nothing
                    fill!(x1dot, 0)
                    if y1dot !== nothing
                        @. x1dot += y1dot*2
                    end
                    if x2dot !== nothing
                        @. x1dot += y2dot*(12*x1^2)
                    end
                end
                if x2dot !== nothing
                    fill!(x2dot, 0)
                    if y1dot !== nothing
                        @. x2dot += y1dot*(6*x2)
                    end
                    if y2dot !== nothing
                        @. x2dot += y2dot*(20*x2^3)
                    end
                end
            end
            return nothing
        end

        p = om.Problem()
        nrows, ncols = 2, 3
        ecomp = ECompMatrixFree(nrows, ncols)
        comp = make_component(ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x1", "x2"], promotes_outputs=["y1", "y2"])
        p.setup(force_alloc_complex=true)
        p.set_val("x1", reshape(0:(nrows*ncols-1), nrows, ncols) .+ 0.5)
        p.set_val("x2", reshape(0:(nrows*ncols-1), nrows, ncols) .+ 1.0)
        p.run_model()

        # Test that the outputs are what we expect.
        expected = 2 .* PyArray(p.get_val("x1")) .+ 3 .* PyArray(p.get_val("x2")).^2
        actual = PyArray(p.get_val("y1"))
        @test expected ≈ actual

        expected = 4 .* PyArray(p.get_val("x1")).^3 .+ 5 .* PyArray(p.get_val("x2")).^4
        actual = PyArray(p.get_val("y2"))
        @test expected ≈ actual

        # Check that partials approximated by the complex-step method match the user-provided partials.
        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
                @test PyArray(cpd_comp_var_wrt["J_rev"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

        p.set_val("x1", reshape(0:(nrows*ncols-1), nrows, ncols) .+ 4.0)
        p.set_val("x2", reshape(0:(nrows*ncols-1), nrows, ncols) .+ 5.0)

        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
                @test PyArray(cpd_comp_var_wrt["J_rev"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end

    @safetestset "shape_by_conn" begin
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test

        struct ECompShapeByConn <: OpenMDAOCore.AbstractExplicitComp end

        function OpenMDAOCore.setup(self::ECompShapeByConn)
            input_data = [OpenMDAOCore.VarData("x"; shape_by_conn=true)]
            output_data = [OpenMDAOCore.VarData("y"; shape_by_conn=true, copy_shape="x")]

            partials_data = []
            return input_data, output_data, partials_data
        end

        function OpenMDAOCore.setup_partials(self::ECompShapeByConn, input_sizes, output_sizes)
            @assert input_sizes["x"] == output_sizes["y"]
            m, n = input_sizes["x"]
            rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=Dict(:i=>m, :j=>n), of_ss=[:i, :j], wrt_ss=[:i, :j])
            partials_data = [OpenMDAOCore.PartialsData("y", "x"; rows=rows, cols=cols)]

            return partials_data
        end

        function OpenMDAOCore.compute!(self::ECompShapeByConn, inputs, outputs)
            x = inputs["x"]
            y = outputs["y"]
            y .= 2 .* x.^2 .+ 1
            return nothing
        end

        function OpenMDAOCore.compute_partials!(self::ECompShapeByConn, inputs, partials)
            x = inputs["x"]
            m, n = size(x)
            # So, with the way I've declared the partials above, OpenMDAO will have
            # created a Numpy array of shape (m, n) and then flattened it. So, to get
            # that to work, I'll need to do this:
            dydx = PermutedDimsArray(reshape(partials["y", "x"], n, m), (2, 1))
            dydx .= 4 .* x
            return nothing
        end

        m, n = 3, 4
        p = om.Problem()
        comp = om.IndepVarComp()
        comp.add_output("x", shape=(m, n))
        p.model.add_subsystem("inputs_comp", comp, promotes_outputs=["x"])

        ecomp = ECompShapeByConn()
        comp = make_component(ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=true)
        p.set_val("x", 1:m*n)
        p.run_model()

        # Test that the output is what we expect.
        expected = 2 .* PyArray(p.get_val("x")).^2 .+ 1
        actual = PyArray(p.get_val("y"))
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end
end

@safetestset "Implicit Components" begin

    @safetestset "simple" begin
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test


        struct SimpleImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
            n::TI  # these would be like "options" in openmdao
            a::TF
        end

        function OpenMDAOCore.setup(self::SimpleImplicit)
         
            n = self.n
            inputs = [
                OpenMDAOCore.VarData("x"; shape=n, val=2.0),
                OpenMDAOCore.VarData("y"; shape=(n,), val=3.0)]

            outputs = [
                OpenMDAOCore.VarData("z1"; shape=(n,), val=fill(2.0, n)),
                OpenMDAOCore.VarData("z2"; shape=n, val=3.0)]

            rows = 0:n-1
            cols = 0:n-1
            partials = [
                OpenMDAOCore.PartialsData("z1", "x"; rows=rows, cols=cols),
                OpenMDAOCore.PartialsData("z1", "y"; rows, cols),
                OpenMDAOCore.PartialsData("z1", "z1"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "x"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "y"; rows, cols),          
                OpenMDAOCore.PartialsData("z2", "z2"; rows, cols)
            ]

            return inputs, outputs, partials
        end

        function OpenMDAOCore.apply_nonlinear!(self::SimpleImplicit, inputs, outputs, residuals)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
            @. residuals["z2"] = (a*x + y) - outputs["z2"]

            return nothing
        end

        function OpenMDAOCore.linearize!(self::SimpleImplicit, inputs, outputs, partials)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. partials["z1", "z1"] = -1.0
            @. partials["z1", "x"] = 2*a*x
            @. partials["z1", "y"] = 2*y

            @. partials["z2", "z2"] = -1.0
            @. partials["z2", "x"] = a
            @. partials["z2", "y"] = 1.0

            return nothing
        end

        p = om.Problem()
        n = 10
        a = 3.0
        icomp = SimpleImplicit(n, a)
        comp = make_component(icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=true)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=true, iprint=2, err_on_non_converge=true)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=true)
        p.set_val("x", 3.0)
        p.set_val("y", 4.0)
        p.run_model()

        # Check outputs.
        expected = a.*PyArray(p.get_val("x")).^2 .+ PyArray(p.get_val("y")).^2
        actual = PyArray(p.get_val("z1"))
        @test actual ≈ expected

        expected = a.*PyArray(p.get_val("x")) .+ PyArray(p.get_val("y"))
        actual = PyArray(p.get_val("z2"))
        @test actual ≈ expected

        # Check partials.
        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that the partials the user provided are correct.
        icomp_partials = pyconvert(Dict, cpd["icomp"])

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "x"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 2 .* a .* PyArray(p.get_val("x"))
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "y"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 2 .* PyArray(p.get_val("y"))
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "z1"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "x"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= a
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "y"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 1.0
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "z2"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end

    @safetestset "solve nonlinear" begin
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test


        struct SolveNonlinearImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
            n::TI  # these would be like "options" in openmdao
            a::TF
        end

        function OpenMDAOCore.setup(self::SolveNonlinearImplicit)
         
            n = self.n
            inputs = [
                OpenMDAOCore.VarData("x"; shape=n, val=2.0),
                OpenMDAOCore.VarData("y"; shape=(n,), val=3.0)]

            outputs = [
                OpenMDAOCore.VarData("z1"; shape=(n,), val=fill(2.0, n)),
                OpenMDAOCore.VarData("z2"; shape=n, val=3.0)]

            rows = 0:n-1
            cols = 0:n-1
            partials = [
                OpenMDAOCore.PartialsData("z1", "x"; rows=rows, cols=cols),
                OpenMDAOCore.PartialsData("z1", "y"; rows, cols),
                OpenMDAOCore.PartialsData("z1", "z1"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "x"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "y"; rows, cols),          
                OpenMDAOCore.PartialsData("z2", "z2"; rows, cols)
            ]

            return inputs, outputs, partials
        end

        function OpenMDAOCore.apply_nonlinear!(self::SolveNonlinearImplicit, inputs, outputs, residuals)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
            @. residuals["z2"] = (a*x + y) - outputs["z2"]

            return nothing
        end

        function OpenMDAOCore.solve_nonlinear!(self::SolveNonlinearImplicit, inputs, outputs)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. outputs["z1"] = a*x*x + y*y
            @. outputs["z2"] = a*x + y

            return nothing
        end

        function OpenMDAOCore.linearize!(self::SolveNonlinearImplicit, inputs, outputs, partials)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. partials["z1", "z1"] = -1.0
            @. partials["z1", "x"] = 2*a*x
            @. partials["z1", "y"] = 2*y

            @. partials["z2", "z2"] = -1.0
            @. partials["z2", "x"] = a
            @. partials["z2", "y"] = 1.0

            return nothing
        end

        p = om.Problem()
        n = 10
        a = 3.0
        icomp = SolveNonlinearImplicit(n, a)
        comp = make_component(icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=true)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=true)
        p.set_val("x", 3.0)
        p.set_val("y", 4.0)
        p.run_model()

        # Check outputs.
        expected = a.*PyArray(p.get_val("x")).^2 .+ PyArray(p.get_val("y")).^2
        actual = PyArray(p.get_val("z1"))
        @test actual ≈ expected

        # Check partials.
        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that the partials the user provided are correct.
        icomp_partials = pyconvert(Dict, cpd["icomp"])

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "x"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 2 .* a .* PyArray(p.get_val("x"))
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "y"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 2 .* PyArray(p.get_val("y"))
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "z1"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "x"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= a
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "y"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 1.0
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "z2"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end

    @safetestset "matrix-free" begin
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test


        struct MatrixFreeImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
            n::TI  # these would be like "options" in openmdao
            a::TF
        end

        function OpenMDAOCore.setup(self::MatrixFreeImplicit)
         
            n = self.n
            inputs = [
                OpenMDAOCore.VarData("x"; shape=n, val=2.0),
                OpenMDAOCore.VarData("y"; shape=(n,), val=3.0)]

            outputs = [
                OpenMDAOCore.VarData("z1"; shape=(n,), val=fill(2.0, n)),
                OpenMDAOCore.VarData("z2"; shape=n, val=3.0)]

            rows = 0:n-1
            cols = 0:n-1
            partials = [
                OpenMDAOCore.PartialsData("z1", "x"; rows=rows, cols=cols),
                OpenMDAOCore.PartialsData("z1", "y"; rows, cols),
                OpenMDAOCore.PartialsData("z1", "z1"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "x"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "y"; rows, cols),          
                OpenMDAOCore.PartialsData("z2", "z2"; rows, cols)]

            return inputs, outputs, partials
        end

        function OpenMDAOCore.apply_nonlinear!(self::MatrixFreeImplicit, inputs, outputs, residuals)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
            @. residuals["z2"] = (a*x + y) - outputs["z2"]

            return nothing
        end

        function OpenMDAOCore.apply_linear!(self::MatrixFreeImplicit, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
            a = self.a
            x, y = inputs["x"], inputs["y"]
            z1, z2 = outputs["z1"], outputs["z2"]

            xdot = get(d_inputs, "x", nothing)
            ydot = get(d_inputs, "y", nothing)
            z1dot = get(d_outputs, "z1", nothing)
            z2dot = get(d_outputs, "z2", nothing)
            Rz1dot = get(d_residuals, "z1", nothing)
            Rz2dot = get(d_residuals, "z2", nothing)

            if mode == "fwd"
                # In forward mode, the goal is to calculate the derivatives of the
                # residuals wrt an upstream input, given the inputs and outputs and the
                # derivatives of the inputs and outputs wrt the upstream input.
                if Rz1dot !== nothing
                    fill!(Rz1dot, 0)
                    if xdot !== nothing
                        @. Rz1dot += 2*a*x*xdot
                    end
                    if ydot !== nothing
                        @. Rz1dot += 2*y*ydot
                    end
                    if z1dot !== nothing
                        @. Rz1dot += -z1dot
                    end
                end
                if Rz2dot !== nothing
                    fill!(Rz2dot, 0)
                    if xdot !== nothing
                        @. Rz2dot += a*xdot
                    end
                    if ydot !== nothing
                        @. Rz2dot += ydot
                    end
                    if z2dot !== nothing
                        @. Rz2dot += -z2dot
                    end
                end
            elseif mode == "rev"
                # In reverse mode, the goal is to calculate the derivatives of an
                # downstream output wrt the inputs and outputs, given the derivatives of
                # the downstream output wrt the residuals.
                if xdot !== nothing
                    fill!(xdot, 0)
                    if Rz1dot !== nothing
                        @. xdot += 2*a*x*Rz1dot
                    end
                    if Rz2dot !== nothing
                        @. xdot += a*Rz2dot
                    end
                end
                if ydot !== nothing
                    fill!(ydot, 0)
                    if Rz1dot !== nothing
                        @. ydot += 2*y*Rz1dot
                    end
                    if Rz2dot !== nothing
                        @. ydot += Rz2dot
                    end
                end
                if z1dot !== nothing
                    fill!(z1dot, 0)
                    if Rz1dot !== nothing
                        @. z1dot += -Rz1dot
                    end
                end
                if z2dot !== nothing
                    fill!(z2dot, 0)
                    if Rz2dot !== nothing
                        @. z2dot += -Rz2dot
                    end
                end
            end
        end

        p = om.Problem()
        n = 10
        a = 3.0
        icomp = MatrixFreeImplicit(n, a)
        comp = make_component(icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=false)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=true, iprint=2, err_on_non_converge=true)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
        p.setup(force_alloc_complex=true)
        p.set_val("x", (0:n-1) .+ 0.5)
        p.set_val("y", (0:n-1) .+ 2)
        p.run_model()

        # Check outputs.
        expected = a.*PyArray(p.get_val("x")).^2 .+ PyArray(p.get_val("y")).^2
        actual = PyArray(p.get_val("z1"))
        @test actual ≈ expected

        expected = a.*PyArray(p.get_val("x")) .+ PyArray(p.get_val("y"))
        actual = PyArray(p.get_val("z2"))
        @test actual ≈ expected

        # Check that the partials the user provided are correct.
        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))
        icomp_partials = pyconvert(Dict, cpd["icomp"])

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "x"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(1:n, 1:n)] .= 2 .* a .* PyArray(p.get_val("x"))
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "y"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected)...)] .= 2 .* PyArray(p.get_val("y"))
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "z1"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "x"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= a
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "y"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 1.0
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "z2"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
                @test PyArray(cpd_comp_var_wrt["J_rev"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end

    @safetestset "solve linear" begin
        using LinearAlgebra: lu, ldiv!
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test

        struct SolveLinearImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
            n::TI  # these would be like "options" in openmdao
            a::TF
        end

        function OpenMDAOCore.setup(self::SolveLinearImplicit)
         
            n = self.n
            inputs = [
                OpenMDAOCore.VarData("x"; shape=n, val=2.0),
                OpenMDAOCore.VarData("y"; shape=(n,), val=3.0)]

            outputs = [
                OpenMDAOCore.VarData("z1"; shape=(n,), val=fill(2.0, n)),
                OpenMDAOCore.VarData("z2"; shape=n, val=3.0)]

            rows = 0:n-1
            cols = 0:n-1
            partials = [
                OpenMDAOCore.PartialsData("z1", "x"; rows=rows, cols=cols),
                OpenMDAOCore.PartialsData("z1", "y"; rows, cols),
                OpenMDAOCore.PartialsData("z1", "z1"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "x"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "y"; rows, cols),          
                OpenMDAOCore.PartialsData("z2", "z2"; rows, cols)
            ]

            return inputs, outputs, partials
        end

        function OpenMDAOCore.apply_nonlinear!(self::SolveLinearImplicit, inputs, outputs, residuals)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
            @. residuals["z2"] = (a*x + y) - outputs["z2"]

            return nothing
        end

        function OpenMDAOCore.solve_nonlinear!(self::SolveLinearImplicit, inputs, outputs)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. outputs["z1"] = a*x*x + y*y
            @. outputs["z2"] = a*x + y

            return nothing
        end

        function OpenMDAOCore.linearize!(self::SolveLinearImplicit, inputs, outputs, partials)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. partials["z1", "z1"] = -1.0
            @. partials["z1", "x"] = 2*a*x
            @. partials["z1", "y"] = 2*y

            @. partials["z2", "z2"] = -1.0
            @. partials["z2", "x"] = a
            @. partials["z2", "y"] = 1.0

            return nothing
        end

        function OpenMDAOCore.solve_linear!(self::SolveLinearImplicit, d_outputs, d_residuals, mode)
            n = self.n
            a = self.a

            z1dot = get(d_outputs, "z1", nothing)
            z2dot = get(d_outputs, "z2", nothing)
            Rz1dot = get(d_residuals, "z1", nothing)
            Rz2dot = get(d_residuals, "z2", nothing)

            if mode == "fwd"
                # In forward mode, the goal is to calculate the total derivatives of the
                # implicit outputs wrt an upstream input, given the
                # derivatives of the residuals wrt the upstream input.
                if z1dot !== nothing
                    pRz1_pz1 = zeros(self.n, self.n)
                    for i in 1:n
                        pRz1_pz1[i, i] = -1
                    end
                    pRz1_pz1_lu = lu(pRz1_pz1)
                    # Annoying: z1dot is a PythonCall.PyArray, which isn't a
                    # StridedArray and so can't be used with ldiv! directly.
                    z1dotfoo = Vector{eltype(z1dot)}(undef, size(z1dot))
                    ldiv!(z1dotfoo, pRz1_pz1_lu, Rz1dot)
                    z1dot .= z1dotfoo
                end

                if z2dot !== nothing
                    pRz2_pz2 = zeros(self.n, self.n)
                    for i in 1:n
                        pRz2_pz2[i, i] = -1
                    end
                    z2dotfoo = Vector{eltype(z2dot)}(undef, size(z2dot))
                    # Annoying: z1dot is a PythonCall.PyArray, which isn't a
                    # StridedArray and so can't be used with ldiv! directly.
                    ldiv!(z2dotfoo, lu(pRz2_pz2), Rz2dot)
                    z2dot .= z2dotfoo
                end

            elseif mode == "rev"
                # In reverse mode, the goal is to calculate the total derivatives of a
                # downstream output wrt a residual, given the total derivative of the
                # downstream output wrt the residual.
                if Rz1dot !== nothing
                    pRz1_pz1 = zeros(self.n, self.n)
                    for i in 1:n
                        pRz1_pz1[i, i] = -1
                    end
                    # The partial derivative of z1's residual wrt z1 is diagonal, so
                    # it's equal to it's transpose.
                    # ldiv!(z1dot, pRz1_pz1, Rz1dot)
                    Rz1dotfoo = Vector{eltype(Rz1dot)}(undef, size(Rz1dot))
                    ldiv!(Rz1dotfoo, lu(pRz1_pz1), z1dot)
                    Rz1dot .= Rz1dotfoo
                end
                if Rz2dot != nothing
                    pRz2_pz2 = zeros(self.n, self.n)
                    for i in 1:n
                        pRz2_pz2[i, i] = -1
                    end
                    # The partial derivative of z2's residual wrt z2 is diagonal, so
                    # it's equal to it's transpose.
                    # ldiv!(z2dot, pRz2_pz2, Rz2dot)
                    Rz2dotfoo = Vector{eltype(Rz2dot)}(undef, size(Rz2dot))
                    ldiv!(Rz2dotfoo, lu(pRz2_pz2), z2dot)
                    Rz2dot .= Rz2dotfoo
                end
            end
            return nothing
        end

        n = 10
        a = 3.0

        p_fwd = om.Problem()
        icomp = SolveLinearImplicit(n, a)
        comp = make_component(icomp)
        p_fwd.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])

        p_fwd.setup(force_alloc_complex=true, mode="fwd")
        p_fwd.set_val("x", (0:n-1).+0.5)
        p_fwd.set_val("y", (0:n-1).+2)
        p_fwd.run_model()

        p_rev = om.Problem()
        icomp = SolveLinearImplicit(n, a)
        comp = make_component(icomp)
        p_rev.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])

        p_rev.setup(force_alloc_complex=true, mode="rev")
        p_rev.set_val("x", (0:n-1).+0.5)
        p_rev.set_val("y", (0:n-1).+2)
        p_rev.run_model()

        for p in [p_fwd, p_rev]
            # Check outputs.
            expected = a.*PyArray(p.get_val("x")).^2 .+ PyArray(p.get_val("y")).^2
            actual = PyArray(p.get_val("z1"))
            @test actual ≈ expected

            expected = a.*PyArray(p.get_val("x")) .+ PyArray(p.get_val("y"))
            actual = PyArray(p.get_val("z2"))
            @test actual ≈ expected

            # Check that the partials the user provided are correct.
            cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))
            icomp_partials = pyconvert(Dict, cpd["icomp"])

            actual = PyArray(pyconvert(Dict, icomp_partials["z1", "x"])["J_fwd"])
            expected = zeros(n, n)
            expected[CartesianIndex.(1:n, 1:n)] .= 2 .* a .* PyArray(p.get_val("x"))
            @test actual ≈ expected

            actual = PyArray(pyconvert(Dict, icomp_partials["z1", "y"])["J_fwd"])
            expected = zeros(n, n)
            expected[CartesianIndex.(axes(expected)...)] .= 2 .* PyArray(p.get_val("y"))
            @test actual ≈ expected

            actual = PyArray(pyconvert(Dict, icomp_partials["z1", "z1"])["J_fwd"])
            expected = zeros(n, n)
            expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
            @test actual ≈ expected

            actual = PyArray(pyconvert(Dict, icomp_partials["z2", "x"])["J_fwd"])
            expected = zeros(n, n)
            expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= a
            @test actual ≈ expected

            actual = PyArray(pyconvert(Dict, icomp_partials["z2", "y"])["J_fwd"])
            expected = zeros(n, n)
            expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 1.0
            @test actual ≈ expected

            actual = PyArray(pyconvert(Dict, icomp_partials["z2", "z2"])["J_fwd"])
            expected = zeros(n, n)
            expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
            @test actual ≈ expected

            # Check that partials approximated by the complex-step method match the user-provided partials.
            for comp in keys(cpd)
                for (pyvar, pywrt) in keys(cpd[comp])
                    var = pyconvert(Any, pyvar)
                    wrt = pyconvert(Any, pywrt)
                    cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                    cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                    @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
                end
            end

            # Check that the total derivatives are good.
            ctd = pyconvert(Dict, p.check_totals(of=["z1", "z2"], wrt=["x", "y"], method="cs", compact_print=true, out_stream=nothing))
            for (pyvar, pywrt) in keys(ctd)
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                J_fwd = pyconvert(Dict, ctd[var, wrt])["J_fwd"]
                J_fd = pyconvert(Dict, ctd[var, wrt])["J_fd"]
                @test J_fwd ≈ J_fd
            end
        end

    end

    @safetestset "guess nonlinear" begin
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test

        struct GuessNonlinearImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
            n::TI  # these would be like "options" in openmdao
            xguess::TF
            xlower::TF
            xupper::TF
        end

        function OpenMDAOCore.setup(self::GuessNonlinearImplicit)
            n = self.n
            xlower = self.xlower
            xupper = self.xupper
            inputs = [
                OpenMDAOCore.VarData("a"; shape=n, val=2.0),
                OpenMDAOCore.VarData("b"; shape=(n,), val=3.0),
                OpenMDAOCore.VarData("c"; shape=(n,), val=3.0)]

            outputs = [OpenMDAOCore.VarData("x"; shape=n, val=3.0, lower=xlower, upper=xupper)]

            rows = 0:n-1
            cols = 0:n-1
            partials = [
                OpenMDAOCore.PartialsData("x", "a"; rows=rows, cols=cols),
                OpenMDAOCore.PartialsData("x", "b"; rows, cols),
                OpenMDAOCore.PartialsData("x", "c"; rows, cols),
                OpenMDAOCore.PartialsData("x", "x"; rows, cols),
            ]

            return inputs, outputs, partials
        end

        function OpenMDAOCore.apply_nonlinear!(self::GuessNonlinearImplicit, inputs, outputs, residuals)
            a = inputs["a"]
            b = inputs["b"]
            c = inputs["c"]
            x = outputs["x"]
            Rx = residuals["x"]

            @. Rx = a*x^2 + b*x + c

            return nothing
        end

        function OpenMDAOCore.linearize!(self::GuessNonlinearImplicit, inputs, outputs, partials)
            a = inputs["a"]
            b = inputs["b"]
            c = inputs["c"]
            x = outputs["x"]

            dRx_da = partials["x", "a"]
            dRx_db = partials["x", "b"]
            dRx_dc = partials["x", "c"]
            dRx_dx = partials["x", "x"]

            @. dRx_da = x^2
            @. dRx_db = x
            @. dRx_dc = 1
            @. dRx_dx = 2*a*x + b

            return nothing
        end

        function OpenMDAOCore.guess_nonlinear!(self::GuessNonlinearImplicit, inputs, outputs, residuals)
            @. outputs["x"] = self.xguess
            return nothing
        end

        n = 9

        # Create a component that should find the left root of R(x) = x**2 - 4*x + 3 = (x - 1)*(x - 3) = 0, aka 1.
        # (x - 1)*(x - 3)
        xguess = 1.5
        xlower = -10.0
        xupper = 1.9
        p_leftroot = om.Problem()
        icomp = GuessNonlinearImplicit(n, xguess, xlower, xupper)
        comp = make_component(icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=true)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=true, iprint=2, err_on_non_converge=true)
        comp.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar")
        p_leftroot.model.add_subsystem("icomp", comp, promotes_inputs=["a", "b", "c"], promotes_outputs=["x"])
        p_leftroot.setup(force_alloc_complex=true)
        p_leftroot.set_val("a", 1.0)
        p_leftroot.set_val("b", -4.0)
        p_leftroot.set_val("c", 3.0)
        p_leftroot.run_model()

        # Create a component that should find the right root of R(x) = x**2 - 4*x + 3 = 0, aka 3.
        xguess = 2.5
        xlower = 2.1
        xupper = 10.0
        p_rightroot = om.Problem()
        icomp = GuessNonlinearImplicit(n, xguess, xlower, xupper)
        comp = make_component(icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=true)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=true, iprint=2, err_on_non_converge=true)
        comp.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement="scalar")
        p_rightroot.model.add_subsystem("icomp", comp, promotes_inputs=["a", "b", "c"], promotes_outputs=["x"])
        p_rightroot.setup(force_alloc_complex=true)
        p_rightroot.set_val("a", 1.0)
        p_rightroot.set_val("b", -4.0)
        p_rightroot.set_val("c", 3.0)
        p_rightroot.run_model()

        # Test that the results are what we expect.
        expected = 1 .* ones(n)
        actual = PyArray(p_leftroot.get_val("x"))
        @test actual ≈ expected

        expected = 3 .* ones(n)
        actual = PyArray(p_rightroot.get_val("x"))
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for p in [p_leftroot, p_rightroot]
            cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))
            for comp in keys(cpd)
                for (pyvar, pywrt) in keys(cpd[comp])
                    var = pyconvert(Any, pyvar)
                    wrt = pyconvert(Any, pywrt)
                    cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                    cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                    @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
                end
            end
        end
    end

    @safetestset "shape_by_conn" begin
        using OpenMDAO
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test

        struct ImplicitShapeByConn{TF} <: OpenMDAOCore.AbstractImplicitComp
            a::TF
        end

        function OpenMDAOCore.setup(self::ImplicitShapeByConn)
            inputs = [
                OpenMDAOCore.VarData("x"; val=2.0, shape_by_conn=true),
                OpenMDAOCore.VarData("y"; val=3.0, shape_by_conn=true, copy_shape="x")]

            outputs = [
                OpenMDAOCore.VarData("z1"; val=2.0, shape_by_conn=true, copy_shape="x"),
                OpenMDAOCore.VarData("z2"; val=3.0, shape_by_conn=true, copy_shape="x")]

            partials = []
            return inputs, outputs, partials
        end

        function OpenMDAOCore.setup_partials(self::ImplicitShapeByConn, input_sizes, output_sizes)
            @assert input_sizes["y"] == input_sizes["x"]
            @assert output_sizes["z1"] == input_sizes["x"]
            @assert output_sizes["z2"] == input_sizes["x"]
            n = only(input_sizes["x"])
            rows = 0:n-1
            cols = 0:n-1
            partials = [
                OpenMDAOCore.PartialsData("z1", "x"; rows=rows, cols=cols),
                OpenMDAOCore.PartialsData("z1", "y"; rows, cols),
                OpenMDAOCore.PartialsData("z1", "z1"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "x"; rows, cols),
                OpenMDAOCore.PartialsData("z2", "y"; rows, cols),          
                OpenMDAOCore.PartialsData("z2", "z2"; rows, cols)
            ]

            return partials
        end

        function OpenMDAOCore.apply_nonlinear!(self::ImplicitShapeByConn, inputs, outputs, residuals)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
            @. residuals["z2"] = (a*x + y) - outputs["z2"]

            return nothing
        end

        function OpenMDAOCore.linearize!(self::ImplicitShapeByConn, inputs, outputs, partials)
            a = self.a
            x = inputs["x"]
            y = inputs["y"]

            @. partials["z1", "z1"] = -1.0
            @. partials["z1", "x"] = 2*a*x
            @. partials["z1", "y"] = 2*y

            @. partials["z2", "z2"] = -1.0
            @. partials["z2", "x"] = a
            @. partials["z2", "y"] = 1.0

            return nothing
        end

        p = om.Problem()
        n = 10
        a = 3.0

        comp = om.IndepVarComp()
        comp.add_output("x", shape=n)
        comp.add_output("y", shape=n)
        p.model.add_subsystem("input_comp", comp, promotes_outputs=["x", "y"])

        icomp = ImplicitShapeByConn(a)
        comp = make_component(icomp)
        comp.linear_solver = om.DirectSolver(assemble_jac=true)
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=true, iprint=2, err_on_non_converge=true)
        p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])

        p.setup(force_alloc_complex=true)
        p.set_val("x", 1:n)
        p.set_val("y", 2:n+1)
        p.run_model()

        # Check outputs.
        expected = a.*PyArray(p.get_val("x")).^2 .+ PyArray(p.get_val("y")).^2
        actual = PyArray(p.get_val("z1"))
        @test actual ≈ expected

        expected = a.*PyArray(p.get_val("x")) .+ PyArray(p.get_val("y"))
        actual = PyArray(p.get_val("z2"))
        @test actual ≈ expected

        # Check partials.
        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that the partials the user provided are correct.
        icomp_partials = pyconvert(Dict, cpd["icomp"])

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "x"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 2 .* a .* PyArray(p.get_val("x"))
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "y"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 2 .* PyArray(p.get_val("y"))
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z1", "z1"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "x"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= a
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "y"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= 1.0
        @test actual ≈ expected

        actual = PyArray(pyconvert(Dict, icomp_partials["z2", "z2"])["J_fwd"])
        expected = zeros(n, n)
        expected[CartesianIndex.(axes(expected, 1), axes(expected, 2))] .= -1.0
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end
    end
end

@safetestset "DymosifiedCompWrapper" begin

    @safetestset "normal operation" begin
        using OpenMDAO: DymosifiedCompWrapper
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test
        
        struct FooODE <: OpenMDAOCore.AbstractExplicitComp
            num_nodes::Int
            arg1::Bool
            arg2::Float64
        end

        FooODE(; num_nodes, arg1, arg2) = FooODE(num_nodes, arg1, arg2)

        arg1 = false
        arg2 = 3.0
        dcw = DymosifiedCompWrapper(FooODE; arg1, arg2)
        @test length(dcw) == 1

        nn = 8
        comp = dcw(num_nodes=nn)
        jlcomp = pyconvert(FooODE, comp.options["jlcomp"])
        @test jlcomp.num_nodes == nn
        @test jlcomp.arg1 == arg1
        @test jlcomp.arg2 == arg2
    end

    @safetestset "require OpenMDAOCore.AbstractComp" begin
        using OpenMDAO: DymosifiedCompWrapper
        using OpenMDAOCore: OpenMDAOCore

        struct FooNonODE
            num_nodes::Int
            arg1::Bool
            arg2::Float64
        end

        FooNonODE(; num_nodes, arg1, arg2) = FooNonODE(num_nodes, arg1, arg2)

        arg1 = false
        arg2 = 3.0
        # This shouldn't because FooNonODE is not an OpenMDAOCore.AbstractComp
        @test_throws MethodError DymosifiedCompWrapper(FooNonODE; arg1, arg2)
    end

    @safetestset "require num_nodes" begin
        using OpenMDAO: DymosifiedCompWrapper
        using OpenMDAOCore: OpenMDAOCore
        using PythonCall
        using Test
        
        struct FooNoNumNodesODE <: OpenMDAOCore.AbstractExplicitComp
            arg1::Bool
            arg2::Float64
        end

        FooNoNumNodesODE(; arg1, arg2) = FooNoNumNodesODE(arg1, arg2)

        arg1 = false
        arg2 = 3.0
        dcw = DymosifiedCompWrapper(FooNoNumNodesODE; arg1, arg2)
        @test length(dcw) == 1

        nn = 8
        # this will fail because FooNoNumNodesODE doesn't have a num_nodes keyword argument
        @test_throws MethodError dcw(num_nodes=nn)
    end
end
