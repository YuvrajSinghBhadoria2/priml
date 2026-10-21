"""Attribute-passthrough mixins for transparent wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Self, cast, overload, override


class PassthroughAttribute[T]:
    """Expose one target attribute without storing a wrapper field."""

    name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        """Store the attribute name in the descriptor."""
        del owner
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...

    @overload
    def __get__(self, instance: ReadPassthroughMixin, owner: type) -> T: ...

    def __get__(
        self,
        instance: ReadPassthroughMixin | None,
        owner: type,
    ) -> Self | T:
        """Get the passthrough attribute value."""
        del owner
        if instance is None:
            return self
        return cast(T, instance.__getattr__(self.name))


class ReadPassthroughMixin:
    """Delegate missing attribute reads to configured backing attributes."""

    _passthrough: ClassVar[str] = ""

    def __init_subclass__(
        cls,
        *,
        passthrough: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the subclass with passthrough configuration."""
        super().__init_subclass__(**kwargs)
        if passthrough is not None:
            cls._passthrough = passthrough

    def _passthrough_target(self, attribute: str) -> object:
        """Get the passthrough target object by attribute name."""
        try:
            value: object = cast(object, object.__getattribute__(self, attribute))
            return value
        except AttributeError:
            parent_getattr: object = getattr(super(), "__getattr__", None)
            if parent_getattr is None:
                raise
            return cast(Callable[[str], object], parent_getattr)(attribute)

    def __getattr__(self, name: str) -> object:
        """Get an attribute, delegating to the passthrough target."""
        parent_getattr: object = getattr(super(), "__getattr__", None)
        if parent_getattr is not None:
            try:
                return cast(Callable[[str], object], parent_getattr)(name)
            except AttributeError:
                pass
        try:
            target = self._passthrough_target(self._passthrough)
            value: object = cast(object, getattr(target, name))
            return value
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}.",
            ) from None


class ReadWritePassthroughMixin(ReadPassthroughMixin):
    """Also delegate writes that belong to a configured backing object."""

    @override
    def __setattr__(self, name: str, value: object) -> None:
        target = self.attribute_target(name)
        if target is self:
            super().__setattr__(name, value)
            return
        if target is not None and hasattr(target, name):
            setattr(target, name, value)
            return
        raise AttributeError(
            f"{type(self).__name__!s} passthrough targets have no attribute {name!r}.",
        )

    def attribute_target(self, name: str) -> object:
        """Resolve this wrapper's write recipient, retaining its owned fields.

        Args:
          name: Attribute being assigned.

        Returns:
          target: The eventual recipient, or None when the backing object is absent.

        """
        if name.startswith("_") or name == self._passthrough:
            return self
        try:
            target = self._passthrough_target(self._passthrough)
        except AttributeError:
            return None
        return attribute_target(target, name)


def attribute_target(config: object, name: str) -> object:
    """Resolve the object receiving a normal or transparent attribute write.

    Args:
      config: Object receiving the assignment, possibly a transparent wrapper.
      name: Attribute being assigned; private and backing fields stay local.

    Returns:
      target: The eventual write recipient, or None for an absent backing object.

    """
    if isinstance(config, ReadWritePassthroughMixin):
        return config.attribute_target(name)
    return config
