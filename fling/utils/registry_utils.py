class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There are two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs) -> None:
        super(Registry, self).__init__(*args, **kwargs)
        self.__trace__ = dict()

    def register(self, module_name=None, force_overwrite=False):
        # used as decorator
        def register_fn(fn):
            if module_name is None:
                name = fn.__name__
            else:
                name = module_name
            Registry._register_generic(self, name, fn, force_overwrite)
            return fn

        return register_fn

    @staticmethod
    def _register_generic(module_dict, module_name, module, force_overwrite):
        if not force_overwrite:
            assert module_name not in module_dict, module_name
        module_dict[module_name] = module

    def get(self, module_name):
        return self[module_name]

    def build(self, obj_type: str, *obj_args, **obj_kwargs) -> object:
        try:
            build_fn = self[obj_type]
            return build_fn(*obj_args, **obj_kwargs)
        except Exception as e:
            # get build_fn fail
            if isinstance(e, KeyError):
                raise KeyError("not support buildable-object type: {}".format(obj_type))
            raise e

    def query(self):
        return self.keys()


CLIENT_REGISTRY = Registry()
SERVER_REGISTRY = Registry()
GROUP_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
DATASET_REGISTRY = Registry()
