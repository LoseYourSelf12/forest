class DetectorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name, detector_cls):
        cls._registry[name] = detector_cls

    @classmethod
    def create(cls, name, *args, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Detector '{name}' not registered")
        return cls._registry[name](*args, **kwargs)


def register_detector(name):
    def decorator(cls):
        DetectorRegistry.register(name, cls)
        return cls
    return decorator
