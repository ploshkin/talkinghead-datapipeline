from typing import Any, Dict


class NodeRegistry(type):

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls) -> Dict[str, Any]:
        return dict(cls.REGISTRY)


def get_node_classes() -> Dict[str, Any]:
    return NodeRegistry.get_registry()
