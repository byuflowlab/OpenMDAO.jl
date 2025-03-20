using OpenMDAOCore: OpenMDAOCore
using ComponentArrays: ComponentVector
using Test

function test1()
    for val_in_cv in [true, false]
        for val_in_add in [true, false]
            for val_in_amd in [true, false]
                for shape_in_add in [true, false]
                    for units_in_cv in [true, false]
                        for units_in_add in [true, false]
                            for units_in_amd in [true, false]
                                # @show val_in_cv val_in_add val_in_amd shape_in_add units_in_cv units_in_add units_in_amd

                                if val_in_cv
                                    X_ca = ComponentVector{Float64}(a=[-3.0, -4.0])
                                else
                                    X_ca = ComponentVector{Float64}()
                                end

                                if units_in_cv
                                    units_dict = Dict{Symbol,String}(:a=>"inch")
                                else
                                    units_dict = Dict{Symbol,String}()
                                end

                                aviary_input_vars = Dict(:a=>Dict{String,Any}("name"=>"foo:bar:baz:a"))
                                if val_in_add
                                    aviary_input_vars[:a]["val"] = 4.0
                                end

                                if shape_in_add
                                    aviary_input_vars[:a]["shape"] = (3, 4)
                                end

                                if units_in_add
                                    aviary_input_vars[:a]["units"] = "ft"
                                end

                                aviary_meta_data = Dict("foo:bar:baz:a"=>Dict{String,Any}())
                                if val_in_amd
                                    aviary_meta_data["foo:bar:baz:a"]["default_value"] = -6.0
                                end

                                if units_in_amd
                                    aviary_meta_data["foo:bar:baz:a"]["units"] = "m"
                                end

                                X_ca_full, units_dict_full, aviary_input_names = OpenMDAOCore._process_aviary_metadata(X_ca, units_dict, aviary_input_vars, aviary_meta_data)

                                if val_in_cv

                                    if units_in_cv
                                        # units in cv always take priority, so no conversion necessary.
                                        factor = 1.0
                                        units_expected = units_dict[:a]
                                    elseif units_in_add
                                        # we're not taking the value from add, so no conversion necessary.
                                        factor = 1.0
                                        units_expected = aviary_input_vars[:a]["units"]
                                    elseif units_in_amd
                                        # we're not taking the value from the amd, so no conversion necessary.
                                        factor = 1.0
                                        units_expected = aviary_meta_data[aviary_input_names[:a]]["units"]
                                    else
                                        # no units at all, so no conversion
                                        factor = 1.0
                                        units_expected = nothing
                                    end
                                    val_expected = [-3.0, -4.0] .* factor
                                    # @test all(X_ca_full[:a] .≈ val_expected)
                                    # @test units_dict_full[:a] == units_expected
                                elseif val_in_add
                                    if units_in_cv

                                        # Units in CV always take priority.
                                        units_expected = units_dict[:a]
                                        if units_in_add
                                            # Need to convert from ft to inches
                                            factor = 12.0
                                        elseif units_in_amd
                                            # Taking value from add without units, but there are default units in the metadata.
                                            # So we need to convert from meters to inches.
                                            factor = 1/0.0254
                                        else
                                            # Only units in cv, so no conversion necessary.
                                            factor = 1.0
                                        end

                                    elseif units_in_add
                                        # Value coming from add, units are in add, no conversion necessary.
                                        units_expected = aviary_input_vars[:a]["units"]
                                        factor = 1.0
                                    elseif units_in_amd
                                        # Value coming from add, units are in amd and nowhere else.
                                        # No conversion necessary.
                                        factor = 1.0
                                        units_expected = aviary_meta_data[aviary_input_names[:a]]["units"]
                                    else
                                        # No units at all, no units necessary.
                                        factor = 1.0
                                        units_expected = nothing
                                    end

                                    if shape_in_add
                                        val_expected = fill(4.0*factor, (3, 4))
                                    else
                                        val_expected = 4.0*factor
                                    end

                                elseif val_in_amd
                                    if units_in_cv
                                        # Units in CV always take priority.
                                        units_expected = units_dict[:a]
                                        if units_in_amd
                                            # Need to convert from meters to inches.
                                            factor = 1/0.0254
                                        elseif units_in_add
                                            # Units are in add, but no units in amd.
                                            # And we have units in CV, so, no conversion.
                                            factor = 1.0
                                        else
                                            # No units in amd or add, so again just use the units in the cv, no conversion necessary.
                                            factor = 1.0
                                        end
                                    elseif units_in_add
                                        # Units in add take priority over those in amd.
                                        units_expected = aviary_input_vars[:a]["units"]
                                        # So we need to convert if amd has units.
                                        if units_in_amd
                                            # Need to convert from meters to ft.
                                            factor = (1/0.0254)*(1/12.0)
                                        else
                                            # No units in amd, so no conversion necessary.
                                            factor = 1.0
                                        end
                                    elseif units_in_amd
                                        units_expected = aviary_meta_data[aviary_input_names[:a]]["units"]
                                        # No units in cv or add, but we have units in md.
                                        # So no conversion necessary.
                                        factor = 1.0
                                    else
                                        units_expected = nothing
                                        # No units at all, so no conversion necessary.
                                        factor = 1.0
                                    end

                                    if shape_in_add
                                        val_expected = fill(-6.0*factor, (3, 4))
                                    else
                                        val_expected = -6.0*factor
                                    end
                                else
                                    # Units won't matter if things are zero, of course.
                                    if shape_in_add
                                        val_expected = fill(0.0, (3, 4))
                                    else
                                        val_expected = 0.0
                                    end
                                    if units_in_cv
                                        units_expected = units_dict[:a]
                                    elseif units_in_add
                                        units_expected = aviary_input_vars[:a]["units"]
                                    elseif units_in_amd
                                        units_expected = aviary_meta_data[aviary_input_names[:a]]["units"]
                                    else
                                        units_expected = nothing
                                    end
                                end

                                @test aviary_input_names[:a] == "foo:bar:baz:a"

                                if units_expected === nothing
                                    @test !(:a in keys(units_dict_full))
                                else
                                    @test units_dict_full[:a] == units_expected
                                end

                                @test val_expected ≈ X_ca_full[:a]
                            end
                        end
                    end
                end
            end
        end
    end
end
test1()
