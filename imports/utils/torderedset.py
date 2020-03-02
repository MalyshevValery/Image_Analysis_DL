"""Typed ordered set (stub)"""
from typing import TypeVar, Generic, Iterable, Dict, List, Union, Tuple, \
    Iterator, Sequence, Set, AbstractSet

from ordered_set import OrderedSet

T = TypeVar('T')
_ItemType = Union[int, slice, Sequence[int]]


class TOrderedSet(OrderedSet, Generic[T]):
    """Typed ordered set"""

    def __init__(self, iterable: Iterable[T] = None) -> None:
        self.map: Dict[T, int] = {}
        self.items: List[T] = []
        super().__init__(iterable)

    def __len__(self) -> int:
        return super().__len__()  # type: ignore

    def __getitem__(self, index: _ItemType) -> 'TOrderedSet[T]':
        return super().__getitem__(index)  # type: ignore

    def copy(self) -> 'TOrderedSet[T]':
        """Return a shallow copy of this object."""
        return super().copy()  # type: ignore

    def __getstate__(self) -> Union[Tuple[None], List[T]]:
        return super().__getstate__()  # type: ignore

    def __setstate__(self, state: Union[Tuple[None], List[T]]) -> None:
        super().__setstate__(state)  # type: ignore

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key)  # type: ignore

    def add(self, key: T) -> None:
        """
        Add `key` as an item to this OrderedSet, then return its index. If `key`
        is already in the OrderedSet, return the index it already had.
        """
        return super().add(key)  # type: ignore

    append = add

    def update(self, sequence: Sequence[T]) -> int:
        """
        Update the set with the given iterable sequence, then return the index
        of the last element inserted.
        """
        return super().update(sequence)  # type: ignore

    def index(self, key: Union[T, Sequence[T]]) -> int:
        """
        Get the index of a given entry, raising an IndexError if it's not
        present. `key` can be an iterable of entries that is not a string, in
        which case this returns a list of indices.
        """
        return super().index(key)  # type: ignore

    def pop(self) -> T:
        """
        Remove and return the last element from the set. Raises KeyError if the
        set is empty.
        """
        return super().pop()  # type: ignore

    def discard(self, key: T) -> None:
        """Remove an element."""
        super().discard(key)

    def clear(self) -> None:
        """Remove all items from this OrderedSet."""
        super().clear()

    def __iter__(self) -> Iterator[T]:
        return super().__iter__()  # type: ignore

    def __reversed__(self) -> Iterator[T]:
        return super().__reversed__()  # type: ignore

    def __repr__(self) -> str:
        return super().__repr__()  # type: ignore

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other)  # type: ignore

    def union(self, *sets: Sequence[T]) -> 'TOrderedSet[T]':
        """Combines all unique items. Each item order by first appearance."""
        return super().union(*sets)  # type: ignore

    def __and__(self, other: AbstractSet[T]) -> AbstractSet[T]:
        return super().__and__(other)  # type: ignore

    def intersection(self, *sets: Sequence[T]) -> 'TOrderedSet[T]':
        """
        Returns elements in common between all sets. Order is defined only
        by the first set.
        """
        return super().intersection(*sets)  # type: ignore

    def difference(self, *sets: Sequence[T]) -> 'TOrderedSet[T]':
        """Returns all elements that are in this set but not the others."""
        return super().difference(*sets)  # type: ignore

    def issubset(self, other: Union[Set[T], Sequence[T]]) -> bool:
        """Report whether another set contains this set."""
        return super().issubset(other)  # type: ignore

    def issuperset(self, other: Union[Set[T], Sequence[T]]) -> bool:
        """Report whether this set contains another set."""
        return super().issuperset(other)  # type: ignore

    def symmetric_difference(self, other: Sequence[T]) -> 'TOrderedSet[T]':
        """
        Return the symmetric difference of two OrderedSets as a new set.
        That is, the new set will contain all elements that are in exactly
        one of the sets.

        Their order will be preserved, with elements from `self` preceding
        elements from `other`.
        """
        return super().symmetric_difference(other)  # type: ignore

    def difference_update(self, *sets: Sequence[T]) -> None:
        """Update this OrderedSet to remove items from other sets."""
        super().difference_update(*sets)

    def intersection_update(self, other: Sequence[T]) -> None:
        """
        Update this OrderedSet to keep only items in another set, preserving
        their order in this set.
        """
        super().intersection_update(other)

    def symmetric_difference_update(self, other: Sequence[T]) -> None:
        """
        Update this OrderedSet to remove items from another set, then
        add items from the other set that were not present in this set.
        """
        super().symmetric_difference_update(other)
