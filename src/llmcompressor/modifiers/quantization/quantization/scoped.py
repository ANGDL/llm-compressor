"""Reuse module names while matching only a resident calibration working set."""

from weakref import WeakKeyDictionary, ref


class CalibrationModuleIndex:
    """Cache topology names, not schemes, observers, or materialized tensors."""

    def __init__(self, model):
        self.model = ref(model)
        self.names = WeakKeyDictionary(
            (module, (index, name))
            for index, (name, module) in enumerate(model.named_modules())
        )

    def scoped_model(self, model, modules):
        if self.model() is not model or any(
            module not in self.names for module in modules
        ):
            self.__init__(model)
        named_modules = sorted(
            (self.names[module][0], self.names[module][1], module)
            for module in set(modules)
            if module in self.names
        )
        return _CalibrationModelView(
            model, tuple((name, module) for _index, name, module in named_modules)
        )


class _CalibrationModelView:
    def __init__(self, model, named_modules):
        self._model = model
        self._named_modules = named_modules

    def named_modules(self):
        return iter(self._named_modules)

    def __getattr__(self, name):
        return getattr(self._model, name)
