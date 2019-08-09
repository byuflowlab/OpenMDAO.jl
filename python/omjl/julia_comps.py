import openmdao.api as om


class JuliaExplicitComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super(JuliaExplicitComp, self).__init__(**kwargs)

        comp_data = self.options['julia_comp_data']
        self._julia_options = {}
        for option in comp_data.options:
            self.options.declare(option.name,
                                 # types=option.type,
                                 default=option.val)
            self._julia_options[option.name] = option.val

        self.options.update(kwargs)

    def initialize(self):
        self.options.declare('julia_comp_data')

    def setup(self):
        comp_data = self.options['julia_comp_data']
        input_data = comp_data.inputs
        output_data = comp_data.outputs

        for var in input_data:
            self.add_input(var.name,
                           shape=var.shape,
                           val=var.val)

        for var in output_data:
            self.add_output(var.name, shape=var.shape, val=var.val)

        for data in comp_data.partials:
            self.declare_partials(data.of, data.wrt)

    def compute(self, inputs, outputs):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)

        comp_data.compute(self._julia_options, inputs_dict, outputs_dict)
        for k, v in outputs_dict.items():
            outputs[k] = v

    def compute_partials(self, inputs, partials):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)

        partials_dict = {}
        for part_names in comp_data.partials:
            of_wrt = part_names.of, part_names.wrt
            partials_dict[of_wrt] = partials[of_wrt]

        comp_data.compute_partials(self._julia_options, inputs_dict,
                                   partials_dict)
