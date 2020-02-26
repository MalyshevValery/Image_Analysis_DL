"""Typed ordered set (stub)"""
from typing import TypeVar, Generic, Iterable, Dict, List, Union, Tuple, \
    Iterator, Sequence, Set, AbstractSet

from ordered_set import OrderedSet

T = TypeVar('T')


class TOrderedSet(OrderedSet, Generic[T]):
    """Typed ordered set"""

    def __init__(self, iterable: Iterable[T] = None) -> None:
        self.map: Dict[T, int] = {}
        self.items: List[T] = []
        super().__init__(iterable)

    def __len__(self) -> int: ...

    def __getitem__(self, index: Union[
        int, slice, Sequence[int]]) -> 'TOrderedSet[T]': ...

    def copy(self) -> 'TOrderedSet[T]':
        """Return a shallow copy of this object."""
        ...

    def __getstate__(self) -> Union[Tuple[None], List[T]]: ...

    def __setstate__(self, state: Union[Tuple[None], List[T]]) -> None: ...

    def __contains__(self, key: object) -> bool: ...

    def add(self, key: T) -> int:
        """
        Add `key` as an item to this OrderedSet, then return its index. If `key`
        is already in the OrderedSet, return the index it already had.
        """
        ...

    append = add

    def update(self, sequence: Sequence[T]) -> int:
        """
        Update the set with the given iterable sequence, then return the index
        of the last element inserted.
        """
        ...

    def index(self, key: Union[T, Sequence[T]]) -> int:
        """
        Get the index of a given entry, raising an IndexError if it's not
        present. `key` can be an iterable of entries that is not a string, in
        which case this returns a list of indices.
        """
        ...

    def pop(self) -> T:
        """
        Remove and return the last element from the set. Raises KeyError if the
        set is empty.
        """
        ...

    def discard(self, key: T) -> None:
        """Remove an element."""
        ...

    def clear(self) -> None:
        """Remove all items from this OrderedSet."""
        ...

    def __iter__(self) -> Iterator[T]: ...

    def __reversed__(self) -> List[T]: ...

    def __repr__(self) -> str: ...

    def __eq__(self, other: object) -> bool: ...

    def union(self, *sets: Sequence[T]) -> 'TOrderedSet[T]':
        """Combines all unique items. Each item order by first appearance."""
        ...

    def __and__(self, other: AbstractSet[T]) -> 'TOrderedSet[T]': ...

    def intersection(self, *sets: Sequence[T]) -> 'TOrderedSet[T]':
        """
        Returns elements in common between all sets. Order is defined only
        by the first set.
        """
        ...

    def difference(self, *sets: Sequence[T]) -> 'TOrderedSet[T]':
        """Returns all elements that are in this set but not the others."""
        ...

    def issubset(self, other: Union[Set[T], Sequence[T]]) -> bool:
        """Report whether another set contains this set."""
        ...

    def issuperset(self, other: Union[Set[T], Sequence[T]]) -> bool:
        """Report whether this set contains another set."""
        ...

    def symmetric_difference(self, other: Sequence[T]) -> 'TOrderedSet[T]':
        """
        Return the symmetric difference of two OrderedSets as a new set.
        That is, the new set will contain all elements that are in exactly
        one of the sets.

        Their order will be preserved, with elements from `self` preceding
        elements from `other`.
        """
        ...

    def difference_update(self, *sets: Sequence[T]) -> None:
        """Update this OrderedSet to remove items from other sets."""
        ...

    def intersection_update(self, other: Sequence[T]) -> None:
        """
        Update this OrderedSet to keep only items in another set, preserving
        their order in this set.
        """
        ...

    def symmetric_difference_update(self, other: Sequence[T]) -> None:
        """
        Update this OrderedSet to remove items from another set, then
        add items from the other set that were not present in this set.
        """
        ...
