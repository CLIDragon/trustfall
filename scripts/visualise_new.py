from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import timedelta
import numpy as np
import math
from tqdm import tqdm

"""
This is a re-implementation of the visualise algorithm in a clearer way, to 
pave the way for a rust implementation and code review. Outputs should be
exactly equal to those produced by `visualise.py`. 

Most PQR handling is based on the following reasoning:
1. A PQR must follow a YF
2. A PQR must not follow a RNO (as the RNO doesn't reveal information)
3. A PQR must be followed with an AII
"""


class Opid(int):
    def __new__(cls, value, *args, **kwargs):
        # We use -1 as an invalid sentinel value.
        if value == None:
            value = -1
        return super(cls, cls).__new__(cls, value)

    def __repr__(self):
        if int(self) != -1:
            return "Opid(%d)" % int(self)
        else:
            return "None"


def duration_from(value: str) -> np.timedelta64:
    if value.endswith("µs"):
        rem, val = math.modf(float(value.removesuffix("µs")))
        return np.timedelta64(int(val), "us") + duration_from(f"{rem * 1000}ns")
    elif value.endswith("ms"):
        rem, val = math.modf(float(value.removesuffix("ms")))
        return np.timedelta64(int(val), "ms") + duration_from(f"{rem * 1000}µs")
    elif value.endswith("ns"):
        # FIXME: This is discarding values.
        rem, val = math.modf(float(value.removesuffix("ns")))
        return np.timedelta64(int(val), "ns")
    elif value.endswith("s"):
        # This must be last, as it excludes the others.
        rem, val = math.modf(float(value.removesuffix("s")))
        return np.timedelta64(int(val), "s") + duration_from(f"{rem * 1000}ms")
    elif not value:
        # TODO: Should this change in future?
        return np.timedelta64("Nat")


@dataclass
class Op:
    id: Opid
    parent: Opid
    duration: np.timedelta64
    type: str


def seek(start: int, operations: list[Op], initial=False) -> tuple[int, int] | None:
    """
    Look for the next operation that is not part of a callstack,
    counting calls and AdvanceInputIterators.

    :start: is the first index to look from. It does not need to be
        a call.
    :initial: is True if this is the very top-level callstack. At this
        point, the first Call is always RSV, which does not call AII, and
        so there is one less AII than Call.

    Matches patterns of the form:

    C{N}A{N}{.}

    where C is a Call, A is an AdvanceInputIterator, and . is any non C or A
    operation.

    Requires:
        :start: to be a valid index of operations.

    Returns:
        None if the pattern doesn't match, otherwise the number of calls and the
        index of the found element.
    """

    calls = 0
    advances = 0
    for i in range(start, len(operations)):
        op = operations[i]
        if op.type.startswith("Call"):
            if advances != 0:
                return None
            calls += 1
        elif op.type.startswith("AdvanceInputIterator"):
            if advances >= calls:
                return None
            advances += 1
        else:
            if calls != advances and (initial and calls - 1 != advances):
                return None
            return (calls, i)
    else:
        # We reached the end of operations without finding a non-C/A operation.
        return None


def find_callstack_end(start: int, operations: list[Op]) -> tuple[int, int]:
    """Handle a callstack.

    This will not work for the toplevel stack due to different handling required
    for the exit point.

    :start: the index in operations of the first Call in the stack.

    Returns (inside, outside):
        inside: the index (start <= ret < len(operations)) of the first operation
            inside the stack
        outside: the index of (start <= ret < len(operations) the first operation
            outside of the stack in the order (inside, outside).

    Raises an error if the callstack does not end before the final operation (i.e.
    the final operation cannot be part of the callstack).
    """
    # TODO: Test on a stack that has members called later.
    # TODO: Do nested callstacks exist (i.e. a callstack that has its members
    #   accessed after another callstack interrupts?)
    # TODO: Handle PQR
    ret = seek(start, operations)
    assert ret is not None  # (rust): replace with proper error handling.
    calls, i = ret
    first = i

    # The callstack is over when we YF from the last call (or equivalently
    # with an OII).
    # Obviously, this can be extended by AII of the same parent.
    last_call = operations[start + calls - 1].id

    # If the call is a ResolveNeighbors, then it will produce a RNI. The callstack
    # may end when the RNI ends - if it does not end, then AII will be called with
    # the RNI.
    rno = Opid(-1)
    for i in range(i, len(operations)):
        op = operations[i]
        assert not op.type.startswith("Call"), start

        if op.parent == last_call:
            if not (op.type.startswith("YieldFrom") or op.type == "OutputIteratorExhausted"):
                continue

            # RNI has RNO as a parent, IIE is always followed by OIE, RSV
            # is never in a stack.

            n_op = operations[i + 1]
            # Any operation followed by AII is not the end.
            if n_op.type == "AdvanceInputIterator" and n_op.parent == last_call:
                continue

            if op.type == "OutputIteratorExhausted":
                # OutputIteratorExhausted is always final
                i += 1
                break
            elif op.type == "YieldFrom(ResolveNeighborsOuter)":
                # RNO is followed by RNI, OIE, PQR, or Call
                rno = op.id
                # FIXME: Assumes that nested callstacks don't exist.
                if n_op.type.startswith("Call"):
                    i += 1
                    break
            else:
                i += 1
                break
        elif op.parent == rno:
            # Either a OIE or RNI
            assert op.type in ("OutputIteratorExhausted", "YieldFrom(ResolveNeighborsInner)")
            n_op = operations[i + 1]

            if (
                n_op.type in ("OutputIteratorExhausted", "YieldFrom(ResolveNeighborsInner)")
                and n_op.parent == rno
            ):
                # Must be a RNI which is followed by another RNI call.
                continue

            rno = Opid(-1)

            # Any operation followed by AII is not the end.
            if n_op.type == "AdvanceInputIterator" and n_op.parent == last_call:
                continue

            if op.type == "YieldFrom(ResolveNeighborsInner)":
                assert operations[i + 1].type == "YieldInto", i
            else:
                assert op.type == "OutputIteratorExhausted"

            i += 1
            break
    else:
        assert False
    assert i < len(operations)

    return (first, i)


def get_next_op(i: int, operations: list[Op]) -> tuple[list[tuple[int, int, int]], int]:
    """Returns the next operation in the current callstack.

    This completely ignores any created callstacks, and thus usage is only meaningful
    when in the top-level callstack. When calling on the top-level callstack, make
    sure that `i` does not refer to a call inside the initial (CCAA) setup.

    Args:
        `i` is the first call in a stack.

    Returns (stacks, index):
        stacks: list of (first call, first non-C/A operation, last operation)
        index: If there is no next operation, i.e. :i: is the last operation,
            then returns len(operations).
    """
    idx = i + 1
    if idx >= len(operations):
        return [], idx

    # There are cases of multiple callstacks next to each other,
    # for example in `type_requires_more_generic_type_params`.
    op = operations[idx]
    stacks = []
    while op.type.startswith("Call"):
        start = idx
        mid, idx = find_callstack_end(idx, operations)
        stacks.append((start, mid, idx - 1))
        op = operations[idx]
    return stacks, idx


def handle_rni(
    start: int, operations: list[Op], yield_froms, yield_intos, yielded_parent, advance_parents, limits, rno=None
) -> tuple[Opid | None, int]:
    """Handle a RNI operation.

    This function only exists to reduce code duplication.

    :start: The index of the RNI in operations.
    :limits: Inclusive, (start, end) for minimum and maximum accessible values.

    The first return value is the value of yielded_parent. This is necessary to
    return because, unlike the other parameters, it is not mutable. The second
    return value is the index of the last handled value. You must skip past
    this value in any parsing, because all values up until that point (inclusive) have
    been handled and re-parsing them will cause duplication.

    A callstack can never occur because only RNO can be followed by a callstack
    and RNI cannot be followed by a RNO without first encountering a YF (at which
    point the function will return).

    If this function is called when handling a RNO, then pass the id of that RNO
    into the :rno: parameter.
    """
    # TODO: Should this be modified to only handle one RNI for the sake of code
    #  simplicity? We could simply ban mutation of i inside the outer look for a
    #  modest performance hit.
    idx = start
    op = operations[start]
    assert op.type == "YieldFrom(ResolveNeighborsInner)", op

    # There may be any number of following ResolveNeighborsInner,
    # in the standard pattern of AII YF AII YF, but without the AII.
    for j in range(idx + 1, limits[1] + 1):
        if operations[j].type == "YieldFrom(ResolveNeighborsInner)":
            assert operations[j].parent == op.parent
        else:
            # 1 greater than the index of the last RNI
            # FIXME: Replace with just the index of the last RNI.
            idx = j
            break
    else:
        assert limits[1] < len(operations) - 1 and operations[idx + 1].type == "YieldInto"
        for k in range(start, idx):
            yield_intos[operations[idx].id].append(operations[k].id)
            yield_froms[operations[k].id] = None
        if rno is not None:
            yield_intos[operations[idx].id].append(rno)
        return yielded_parent, idx - 1

    # TODO: Test that ProduceQueryResult for ResolveNeighbors works.
    #  This may be difficult to find, because I am unsure any non-trivial
    #  query will end with a ResolveNeighbors.
    assert operations[idx].type in (
        "YieldInto",
        "OutputIteratorExhausted",
        "AdvanceInputIterator",
        "ProduceQueryResult",
    )
    if operations[idx].type in ("YieldInto", "ProduceQueryResult"):
        for k in range(start, idx):
            yield_intos[operations[idx].id].append(operations[k].id)
        if rno is not None:
            yield_intos[operations[idx].id].append(rno)
    elif operations[idx].type == "OutputIteratorExhausted":
        # RNI may be followed by an OIE, as it is the standard pattern of
        # AII YF AII YF, without the AII.
        #
        # The OutputIteratorExhausted must be followed by a YieldInto with a
        # different parent or an AII whose parent is the OIE's grandparent.
        assert idx + 1 > limits[1] or operations[idx + 1].type in ("YieldInto", "AdvanceInputIterator")

        if idx + 1 > limits[1]:
            # TODO: Where does the final operation get assigned to?
            pass
        elif operations[idx + 1].type == "YieldInto":
            assert operations[idx + 1].parent != op.parent
            for k in range(start, idx):
                yield_intos[operations[idx + 1].id].append(operations[k].id)
            if rno is not None:
                yield_intos[operations[idx + 1].id].append(rno)
        else:
            assert operations[idx + 1].parent == operations[int(op.parent) - 1].parent
            # See the AII branch below for an explanation.
            yielded_parent = None

            for k in range(start, idx):
                advance_parents[op.parent].append(operations[k].id)

            if rno is not None:
                advance_parents[op.parent].append(op.id)
    elif operations[idx].type == "AdvanceInputIterator":
        # AdvanceInputIterator. We treat this exactly the same
        # as if we were any other YieldFrom encountering a
        # AdvanceInputIterator. See the YieldFrom(ResolveNeighborsOuter)
        # case for an explanation.
        for k in range(start, idx):
            advance_parents[op.parent].append(operations[k].id)
        if rno is not None:
            advance_parents[op.parent].append(op.id)

        yielded_parent = None

    # Regardless of result, all RNIs need to be added to yield_froms.
    for k in range(start, idx):
        yield_froms[operations[k].id] = None

    # idx is the first operation after, so idx - 1 is the last operation handled.
    return yielded_parent, idx - 1


def handle_stack(start, mid, end, operations: list[Op], yield_intos, yield_froms):
    """Parse a callstack.

    :param start: The index of the first Call.
    :param mid: The index of the first operation that is not a Call or an AII.
    :param end: The index of the last operation inside the stack.
    """
    # print(f"Stack: {operations[start].id} to {operations[end].id}")

    # parent: [YF, YF, ...]
    advance_parents = defaultdict(list)

    # The last valid yield that occurred before a YI.
    # A yield is invalid if it occurs right before a CCAA stack or
    # if it is followed by an AdvanceInputIterator.
    # There can only ever be one because it is impossible to chain
    # YieldFroms with different parents without an intermediate
    # YieldInto or AdvanceInputIterator (as RNI cannot take input,
    # a RSV cannot be followed by a RNI, and all other YFs have AII
    # and/or YI).
    yielded_parent: Opid | None = None

    i = mid
    prev_op = operations[i - 1]
    while i <= end:
        op = operations[i]
        next_op = operations[i + 1]

        if i == end:
            if op.type == "OutputIteratorExhausted":
                # TODO: Re-assign the previous YFs
                break

            assert op.type in (
                "YieldFrom(ResolveProperty)",
                "YieldFrom(ResolveCoercion)",
                "YieldFrom(ResolveNeighborsOuter)",
                "YieldFrom(ResolveNeighborsInner)",  # TODO: Assert must be followed by YI.
            )
            advance_parents[op.parent].append(op.id)

            assert next_op.type in ("YieldInto", "AdvanceInputIterator") or next_op.type.startswith("Call")

            if next_op.type == "YieldInto":
                yield_intos[next_op.id].extend(advance_parents[op.parent])
                del advance_parents[op.parent]
            elif next_op.type == "AdvanceInputIterator":
                # TODO: Finish handling this case. Where should the operations be assigned?
                pass
            elif next_op.type.startswith("Call"):
                # TODO: Finish handling this case. Where should the operations be assigned?
                assert op.type == "YieldFrom(ResolveNeighborsOuter)"

            break

        if op.type == "YieldInto":
            # No adjacent YieldIntos
            assert prev_op.type != "YieldInto" and next_op.type != "YieldInto"

            # YieldInto is always followed by a YieldFrom
            assert next_op.type.startswith("YieldFrom")

            # YieldInto always has the same parent as the YieldFrom that follows it
            assert next_op.parent == op.parent

            yield_froms[next_op.id] = op.id

            # If there is a valid preceding YF, then we want to attach all of its
            # yields to this YI.
            if yielded_parent is not None:
                # This can never be true, as it would require a YF followed by a YI
                # of the same function, without an intermediate AII.
                # Only RSV (which only occurs once) and RNI (which has no YI) can
                # have no parent.
                assert op.parent != yielded_parent

                yield_intos[op.id].extend(advance_parents[yielded_parent])
                del advance_parents[yielded_parent]
                yielded_parent = None
        elif op.type.startswith("YieldFrom"):
            yielded_parent = op.parent

            if op.type == "YieldFrom(ResolveStartingVertices)":
                # YieldFrom(ResolveStartingVertices) must be followed by a YieldInto
                # or a ProduceQueryResult.
                assert next_op.type in ("YieldInto", "ProduceQueryResult")
                yield_intos[next_op.id].append(op.id)

                # ResolveStartingVertices does not have a YieldInto.
                assert op.id not in yield_froms
                yield_froms[op.id] = None
            elif op.type != "YieldFrom(ResolveNeighborsInner)":
                # YieldFrom is always preceded by a YieldInto, unless it is a
                # YieldFrom(ResolveStartingVertices) or a YieldFrom(ResolveNeighborsInner)
                assert prev_op.type == "YieldInto"

            if op.type == "YieldFrom(ResolveNeighborsInner)":
                # ResolveNeighborsInner does not have a YieldInto.
                yielded_parent, skip = handle_rni(
                    i, operations, yield_froms, yield_intos, yielded_parent, advance_parents, limits=(start, end)
                )
                i = skip

            # YieldFrom(ResolveNeighborsOuter) must be followed by RNI or OIE.
            # Outside of the callstack, there may also be a AII.
            if op.type == "YieldFrom(ResolveNeighborsOuter)":
                idx = i + 1
                assert operations[idx].type in (
                    "YieldFrom(ResolveNeighborsInner)",
                    "OutputIteratorExhausted",
                ), (start, end)

                # Because i is always followed by a YieldFrom(ResolveNeighborsInner)
                # or an OutputIteratorExhausted.
                if operations[idx].type == "YieldFrom(ResolveNeighborsInner)":
                    yielded_parent, skip = handle_rni(
                        i + 1,
                        operations,
                        yield_froms,
                        yield_intos,
                        yielded_parent,
                        advance_parents,
                        limits=(start, end),
                        rno=op.id,
                    )
                    i = skip
                else:
                    # Must be an OutputIteratorExhausted.

                    # The OutputIteratorExhausted must be followed by an AdvanceInputIterator
                    # which has the same parent. Then we belong to the AdvanceInputIterator
                    assert operations[idx].type == "OutputIteratorExhausted"
                    assert operations[idx].parent == op.id
                    assert operations[idx + 1].type in (
                        "AdvanceInputIterator",
                        "YieldInto",
                    )

                    if operations[idx + 1].type == "AdvanceInputIterator":
                        # See YieldFrom(ResolveNeighborsOuter) for an explanation
                        # and so on.
                        # assert operations[idx+1].parent == op.parent

                        # This cannot get assigned to an earlier child because seek() guarantees
                        # that the only middle entries are Call and AdvanceInputIterator.
                        advance_parents[op.parent].append(op.id)

                        # The discard is necessary because we set it above.
                        yielded_parent = None
                    else:
                        # Must be YieldInto
                        yield_intos[operations[idx + 1].id].append(op.id)
            elif op.type != "YieldFrom(ResolveNeighborsInner)":
                #
                assert not next_op.type.startswith("YieldFrom")

            # A YieldFrom followed by a YieldInto must have different parents.
            assert not (next_op.type == "YieldInto" and operations[i + 1].parent == op.parent)

            if op.type in (
                "YieldFrom(ResolveProperty)",
                "YieldFrom(ResolveCoercion)",
            ):
                if next_op.type == "YieldInto":
                    yield_intos[next_op.id].append(op.id)
                elif next_op.type == "AdvanceInputIterator":
                    # If the parent is the same, then we are simply repeating
                    # a loop. If the parent is different, then ...
                    # ASSUME if the parent is different, then we were just part
                    # of a big stack following a CCAA call stack.
                    # TODO: Test this assumption.
                    advance_parents[op.parent].append(op.id)

                    # The discard is necessary because we set it above.
                    yielded_parent = None
                elif next_op.type.startswith("ProduceQueryResult"):
                    yield_intos[next_op.id].append(op.id)
        elif op.type.startswith("AdvanceInputIterator"):
            if op.parent in advance_parents:
                yielded_parent = None

            # If AdvanceInputIterator is preceded by YieldFrom, then the parents of the
            # two must be the same.
            if prev_op.type.startswith("YieldFrom"):
                if prev_op.parent == op.parent:
                    # Then the following operations belong to the AdvanceInputIterator
                    # and the AdvanceInputIterator belongs to the next YieldFrom with a
                    # different parent that comes after a YieldFrom of the same parent.
                    pass
                else:
                    # If the preceding yield has a different parent, then assume that
                    # this is the end of a CCAA call stack - since we are already
                    # in a call stack, this should be an unreachable state.
                    assert False
        elif op.type.startswith("ProduceQueryResult"):
            # A ProduceQueryResult must be preceded by a YieldFrom.
            assert operations[i - 1].type.startswith("YieldFrom")
        elif op.type.startswith("Call"):
            assert False, ("Cannot nest callstacks", start, end, i)

        # TODO: Handle skips.
        prev_op = op
        i += 1
    return advance_parents


def parse_file(filename) -> tuple[str, list[Op]]:
    """
    filename: The path to a file that contains a .ptrace.txt format trace output/

    Returns (total_time, operations):

        A list of operations in the file
    """
    with open(filename, encoding="utf-8") as fd:
        lines = fd.readlines()

    total_time = lines[0].strip().split()[1]

    operations = []
    for line in lines[1:]:
        opid, parent, duration, type = line.strip().split(" ", 3)
        opid = Opid(opid[5:-1])
        # Only Call may have invalid parents.
        assert not type.startswith("Call") or parent == "None"
        parent = Opid(None) if parent == "None" else Opid(parent[10:-2])

        assert not type.startswith("YieldFrom") or duration != "None"
        assert not type.startswith("YieldInto") or duration != "None"
        duration = duration_from(duration[5:-1])
        operations.append(Op(opid, parent, duration, type))

    return total_time, operations

from tqdm import tqdm

def calc_yields(operations: list[Op], yield_intos: dict[Opid, list], yield_froms: dict[Opid, Opid | None]):
    """
    Args:
        operations: List of operations that have been executed, ordered by
            Opid.
        yield_intos: A dictionary mapping YI or PQR operations to the YFs that
            constitute them.
        yield_froms: A dictionary mapping each YF to its corresponding YI or
            None if it doesn't have one.
    """
    # parent: [YF, YF, ...]
    # Every operation that is repeated with an AII is placed into this list until
    # it is yielded into a YF.
    advance_parents = defaultdict(list)

    # advance_parents from a callstack.
    # (start, end): advance_parents
    # TODO: Can they be merged into yis in some way?
    dangling_parents = {}

    # The last valid yield that occurred before a YI.
    # A yield is invalid if it occurs right before a CCAA stack or
    # if it is followed by an AdvanceInputIterator.
    # There can only ever be one because it is impossible to chain
    # YieldFroms with different parents without an intermediate
    # YieldInto or AdvanceInputIterator (as RNI cannot take input,
    # a RSV cannot be followed by a RNI, and all other YFs have AII
    # and/or YI).
    yielded_parent: Opid | None = None

    # FIXME: Deduplicate AII handling.
    # Skip the first CCAA setup.
    # SAFETY: A well-formed file will always start with a callstack.
    val = seek(0, operations, initial=True)
    assert val is not None
    _, i = val

    # SAFETY: i is always >= 2, as there is always at least one call at the start
    # of the file. The previous operation is always the last operation to be
    # processed, not the operation before in the list, while next_op is always
    # the next operation in the list not the next operation to be processed.
    # Since calls can only follow RNO, next_op is valid in all cases except for RNO.
    prev_op = operations[i - 1]
    pbar = tqdm(total=len(operations), initial=i)
    while i < len(operations) - 1:
        pbar.update(i - int(prev_op.id) + 1)
        op = operations[i]
        next_op = operations[i + 1]

        # FIXME: Currently assume that the output of a callstack is never
        # re-used.
        # for k, d in dangling_parents.items():
        #     # Are the outputs of callstacks ever re-used?
        #     if op.parent in d and op.type not in ("OutputIteratorExhausted",):
        #         assert False

        if op.type == "YieldInto":
            # No adjacent YieldIntos
            assert prev_op.type != "YieldInto" and next_op.type != "YieldInto"

            # YieldInto is always followed by a YieldFrom
            assert next_op.type.startswith("YieldFrom")

            # YieldInto always has the same parent as the YieldFrom that follows it
            assert next_op.parent == op.parent

            yield_froms[next_op.id] = op.id

            # If there is a valid preceding YF, then we want to attach all of its
            # yields to this YI.
            if yielded_parent is not None:
                # This can never be true, as it would require a YF followed by a YI
                # of the same function, without an intermediate AII.
                # Only RSV (which only occurs once) and RNI (which has no YI) can
                # have no parent.
                assert op.parent != yielded_parent

                yield_intos[op.id].extend(advance_parents[yielded_parent])
                del advance_parents[yielded_parent]
                yielded_parent = None
        elif op.type.startswith("YieldFrom"):
            yielded_parent = op.parent

            if op.type not in ("YieldFrom(ResolveNeighborsInner)", "YieldFrom(ResolveStartingVertices)"):
                # YieldFrom is always preceded by a YieldInto, unless it is a
                # YieldFrom(ResolveStartingVertices) or a YieldFrom(ResolveNeighborsInner)
                assert prev_op.type == "YieldInto"

            if op.type not in ("YieldFrom(ResolveNeighborsInner)", "YieldFrom(ResolveNeighborsOuter)"):
                # Only RNI or RNO may be immediately followed by another YF (and it can only be RNI)
                assert not next_op.type.startswith("YieldFrom")

            # A YieldFrom followed by a YieldInto must have different parents.
            assert not (next_op.type == "YieldInto" and next_op.parent == op.parent)

            if op.type == "YieldFrom(ResolveStartingVertices)":
                # YieldFrom(ResolveStartingVertices) must be followed by a YieldInto
                # or a ProduceQueryResult.
                assert next_op.type in ("YieldInto", "ProduceQueryResult")
                yield_intos[next_op.id].append(op.id)

                # ResolveStartingVertices does not have a YieldInto.
                assert op.id not in yield_froms
                yield_froms[op.id] = None
            elif op.type == "YieldFrom(ResolveNeighborsInner)":
                # ResolveNeighborsInner does not have a YieldInto.
                assert op.id not in yield_froms
                yield_froms[op.id] = None

                yielded_parent, skip = handle_rni(
                    i,
                    operations,
                    yield_froms,
                    yield_intos,
                    yielded_parent,
                    advance_parents,
                    limits=(0, len(operations)),
                )
                i = skip
            elif op.type == "YieldFrom(ResolveNeighborsOuter)":
                if next_op.type.startswith("Call"):
                    # next_op is not correct since it is a call.
                    s, idx = get_next_op(i, operations)

                    assert operations[idx].type in (
                        "AdvanceInputIterator",
                        "YieldInto",
                    )
                    if operations[idx].type == "AdvanceInputIterator":
                        # Handle the same as any AII.
                        assert operations[idx].parent == op.parent
                        yielded_parent = None
                        advance_parents[op.parent].append(op.id)
                    elif operations[idx].type == "YieldInto":
                        yield_intos[operations[idx].id].append(op.id)
                else:
                    # A YF(RNO) is always followed by a YieldFrom(ResolveNeighborsInner)
                    # or an OutputIteratorExhausted if it is not followed by a callstack.
                    assert next_op.type in (
                        "YieldFrom(ResolveNeighborsInner)",
                        "OutputIteratorExhausted",
                    )
                    if next_op.type == "YieldFrom(ResolveNeighborsInner)":
                        yielded_parent, skip = handle_rni(
                            i + 1,
                            operations,
                            yield_froms,
                            yield_intos,
                            yielded_parent,
                            advance_parents,
                            limits=(0, len(operations)),
                            rno=op.id,
                        )
                        i = skip
                    else:
                        # Must be an OutputIteratorExhausted.

                        # The OutputIteratorExhausted must be followed by an AdvanceInputIterator
                        # which has the same parent or a YI. Then we belong to the AdvanceInputIterator/
                        # YieldInto.
                        assert next_op.type == "OutputIteratorExhausted"
                        assert next_op.parent == op.id
                        assert operations[i + 2].type in (
                            "AdvanceInputIterator",
                            "YieldInto",
                        )

                        if operations[i + 2].type == "AdvanceInputIterator":
                            # We assume that if we encounter an AII with the same parent then we are
                            # looping until we find a valid output value.
                            assert operations[i + 2].parent == op.parent

                            # This cannot get assigned to an earlier child because seek() guarantees
                            # that the only middle entries are Call and AdvanceInputIterator.
                            advance_parents[op.parent].append(op.id)

                            # Discard the yielded parent. This is necessary because the parent is set
                            # by default for all YieldFrom operations (at the top of the conditional).
                            yielded_parent = None
                        else:
                            # Must be YieldInto
                            yield_intos[operations[i + 2].id].append(op.id)
            elif op.type in (
                "YieldFrom(ResolveProperty)",
                "YieldFrom(ResolveCoercion)",
            ):
                assert next_op.type in ("YieldInto", "AdvanceInputIterator", "ProduceQueryResult")

                if next_op.type == "YieldInto":
                    yield_intos[next_op.id].append(op.id)
                elif next_op.type == "AdvanceInputIterator":
                    # Handled the same as any AII.
                    assert next_op.parent == op.parent
                    advance_parents[op.parent].append(op.id)
                    yielded_parent = None
                elif next_op.type.startswith("ProduceQueryResult"):
                    yield_intos[next_op.id].append(op.id)
        elif op.type.startswith("AdvanceInputIterator"):
            yielded_parent = None

            # If AdvanceInputIterator is preceded by YieldFrom, then the parents of the
            # two must be the same.
            if prev_op.type.startswith("YieldFrom"):
                if prev_op.parent == op.parent:
                    # Then the following operations belong to the AdvanceInputIterator
                    # and the AdvanceInputIterator belongs to the next YieldFrom with a
                    # different parent that comes after a YieldFrom of the same parent.
                    pass
                else:
                    # If the preceding yield has a different parent, then this would be
                    # the end of a CCAA call stack. This should be unreachable, as
                    # prev_op is callstack aware.
                    assert False
        elif op.type.startswith("ProduceQueryResult"):
            # A ProduceQueryResult must be preceded by a YieldFrom.
            assert prev_op.type.startswith("YieldFrom")
            assert next_op.type == "AdvanceInputIterator"
        elif op.type.startswith("Call"):
            # A callstack may only be started after a
            # YieldFrom(ResolveNeighborsOuter) or Call.
            assert prev_op.type.startswith("YieldFrom(ResolveNeighborsOuter)") or prev_op.type.startswith(
                "Call"
            )

        prev_op = op
        # We cannot just use next_op because i might have been changed inside the loop.
        if operations[i + 1].type.startswith("Call"):
            stacks, i = get_next_op(i, operations)
            for start, mid, end in stacks:
                # We have a stack!
                dangling_parent = handle_stack(start, mid, end, operations, yield_intos, yield_froms)
                dangling_parents[start, end] = dangling_parent
        else:
            i += 1

    assert i + 1 == len(operations)
    if i + 1 == len(operations):
        assert operations[i].type == "OutputIteratorExhausted"

    pbar.close()

    # print("Advance Parents", advance_parents)
    # print("Yield Intos", yield_intos)

    return advance_parents, dangling_parents


import os
import sys

# Test cases:
# enum_discriminants_undefined_non_exhaustive_variant : Call is loooong.
# method_parameter_count_changed : YF (resolve_neighbors_outer)


def format_time(t: np.timedelta64) -> str:
    num, unit = str(t).split(" ")
    if unit == "nanoseconds":
        return f"{num}ns"
    elif unit == "microseconds":
        return f"{num}µs"
    return f"{num} {unit}"


ALLOW_LIST = [
    # "feature_missing",
    # "feature_not_enabled_by_default",
    # "function_export_name_changed",
    # "exported_function_changed_abi",
    "method_parameter_count_changed",
    # "method_requires_different_const_generic_params",
    # "method_requires_different_generic_type_params",
    # "auto_trait_impl_removed",
    # "derive_trait_impl_removed",
    # "sized_impl_removed",
    # "trait_method_added",
    # "trait_method_default_impl_removed",
    # "trait_method_missing",
    # "trait_method_now_doc_hidden",
    # "trait_method_parameter_count_changed",
    # "trait_method_requires_different_const_generic_params",
    # "trait_method_requires_different_generic_type_params",
    # "trait_method_unsafe_added",
    # "sized_impl_removed",
    # "enum_variant_added",
    # "trait_method_marked_deprecated",
    # "enum_discriminants_undefined_non_exhaustive_variant",
]

# Slow lints - all lints that take > 0.4 seconds. Approximately 75% of runtime
# is in these lints.
ALLOW_LIST = [
    "auto_trait_impl_removed",
    "derive_trait_impl_removed",
    "enum_unit_variant_changed_kind",
    "enum_variant_marked_non_exhaustive",
    "enum_variant_missing",
    "inherent_method_missing",
    "inherent_method_now_doc_hidden",
    "inherent_method_unsafe_added",
    "method_parameter_count_changed",
    "method_requires_different_const_generic_params",
    "method_requires_different_generic_type_params",
    "partial_ord_enum_variants_reordered",
    "sized_impl_removed",
    "struct_pub_field_missing",
]

ALLOW_LIST = [x + ".ptrace.txt" for x in ALLOW_LIST]

# TESTS = {}

# TODO: YFs inside callstacks should be counted.
import textwrap

def output_callcounts(yield_froms, operations):
    """Output the callcounts of each function.

    The purpose of this format is to find potential performance improvements due
    to irrelevant calls being processed.
    """
    # TODO: Callcounts extracted from YFs are not a good measure of actual callcounts.
    # A new metric is needed.
    self_times = {}
    for yf, yi in yield_froms.items():
        if yi == None:
            self_times[operations[int(yf) - 1].id] = operations[int(yf) - 1].duration
        else:
            self_times[operations[int(yf) - 1].id] = (
                operations[int(yf) - 1].duration - operations[int(yi) - 1].duration
            )

    # Dictionary mapping each unique callsignature to a list of times.
    # All parts of the call signature are important - you cannot e.g. remove the Vid.
    parents = defaultdict(list)
    # Dictionary mapping each callsignature (of a RNO) to a dictionary of RNIs, each
    # of which contains the list of times for each yield.
    resolve_neighbors = defaultdict(lambda: defaultdict(list))
    for yf, time in self_times.items():
        op = operations[int(yf) - 1]
        parent = operations[int(op.parent) - 1]

        # Separate RNO
        if parent.type == "YieldFrom(ResolveNeighborsOuter)":
            grandparent = operations[int(parent.parent) - 1]
            resolve_neighbors[grandparent.type[5:-1]][parent.id].append(time)
        elif parent.type.startswith("Call"):
            parents[parent.type[5:-1]].append(time)
        else:
            print(parent)

    # print(resolve_neighbors.keys())

    ret = []
    for parent, times in parents.items():
        # Remove parents with a time of 1. This is just to remove noise; queries with
        # only a single call are not going to dominate runtime.
        # if times == 1:
        #     continue

        opid = None
        for op in operations:
            if parent in op.type:
                opid = op.id
                break
            elif not op.type.startswith("Call"):
                break
        ret.append((opid, parent, len(times)))

    for opid, signature, times in ret:
        # parent = operations[int(o) - 1].id

        # Here, times is the total number of times YF (function) are called. This should
        # be the same as the number of YI (function). Note this is not the same as the
        # number of calls.
        print(f"{opid} {signature} [{times}]", end="")

        if signature in resolve_neighbors:
            inner_yields = []
            for k, v in sorted((resolve_neighbors[signature].items()), key=lambda x: int(x[0])):
                # print("K:", type(k))
                # print("-", k, operations[int(operations[int(k) - 1].parent - 1)].type, len(v))
                inner_yields.append(len(v))

            # yielded_rnis is the number of RNIs that have at least one yield in the output - i.e.
            # times - calls is the number of RNIS with OIE.
            yield_counts = Counter(inner_yields)
            yielded_rnis = len(inner_yields)
            empty_yields = times - yielded_rnis
            if empty_yields > 0:
                yield_counts[0] = empty_yields
            if len(yield_counts.most_common(2)) == 1 and yield_counts.most_common(1)[0][0] == 1:
                # Each RNI outputs only a single element.
                print(f" <1:1>")
            else:
                print()
                print("-", "\n".join(textwrap.wrap(str(sorted(yield_counts.items(), key=lambda x: x[0])), width=80)))
                total = sum(inner_yields)
                print("-", f"total: {total}")
        else:
            print()


for entry in os.scandir(R"C:\Users\josep\dev\gsoc\cargo\trustfall\scripts\outputs_datazone"):
    if not entry.is_file():
        continue
    if entry.name not in ALLOW_LIST:
        continue
    # if entry.name not in ("constructible_struct_adds_field.ptrace.txt",):
    #     continue
    # if entry.name < "method_parameter_count_changed.ron.ptrace.txt":
    #     continue

    if not entry.path.endswith(".ptrace.txt"):
        continue

    # TODO: How can we visualise whether an operation is part of a callstack?
    print(f"=== {entry.name.split('.')[0]} === ")
    total_time, operations = parse_file(entry.path)

    yield_intos = defaultdict(list)
    yield_froms = {}
    advance_parents, dangling_parents = calc_yields(operations, yield_intos, yield_froms)

    # TESTS[entry.name] = (yield_intos, yield_froms, advance_parents, operations, total_time)
    # assert TESTS[entry.name] == (yield_intos, yield_froms, advance_parents, operations, total_time), entry.name

    # continue

    if len(sys.argv) == 1:
        continue
    if sys.argv[1] == "yi":
        print("*Yield Intos*")
        for k, v in yield_intos.items():
            print(k, v, sep=" : ")

        print("*Advance Parents*")
        for k, v in advance_parents.items():
            print(k, v, sep=" : ")

        print("*Dangling Parents*")
        for k, dangling_parent in dangling_parents.items():
            print(f"<{k[0]}, {k[1]}>", end=" ")
            if not dangling_parent:
                print("---")
            for k, v in dangling_parent.items():
                print(k, v, sep=" : ")
    elif sys.argv[1] == "call":
        output_callcounts(yield_froms, operations)

    if sys.argv[1] != "yf":
        continue

    print(f"Time: {total_time}")

    overhead_offset = 100  # ns
    ignore_overhead = False  # TODO: Implement this.
    print_neighbors = False

    # How much self-time was spent in each YF?
    # YF(id) : self._time
    self_times = {}
    for yf, yi in yield_froms.items():
        if yi == None:
            self_times[operations[int(yf) - 1].id] = operations[int(yf) - 1].duration
        else:
            self_times[operations[int(yf) - 1].id] = (
                operations[int(yf) - 1].duration - operations[int(yi) - 1].duration
            )

    # print(self_times)

    # Work out how much time was spent in each function call (modeled by Parent)
    # (C/RNO)(id) : list[times]
    parents = defaultdict(list)
    resolve_neighbors = defaultdict(lambda: defaultdict(list))
    for yf, time in self_times.items():
        op = operations[int(yf) - 1]
        parent = operations[int(op.parent) - 1]

        # Separate RNO
        if parent.type == "YieldFrom(ResolveNeighborsOuter)":
            resolve_neighbors[operations[int(parent.id) - 1].parent][parent.id].append(time)
        else:
            parents[parent.id].append(time)

    # print(resolve_neighbors)

    # Collapse parents with the same name.
    n_parents = defaultdict(list)
    for parent, times in parents.items():
        par = operations[int(parent) - 1]

        # Remove the Vid.
        # This is a bad idea because often calls are heterogenous across
        # Vids and we lose valuable information by discarding it!
        # "I can see the instances of name resolution and path resolution do have different
        # means, and there aren't overly many calls even for the size of `aws-sdk-ec2`"
        # n_parents[re.sub(r"Vid\(.*?\), ", "", par.type[5:-1])].extend(times)
        n_parents[par.type[5:-1]].extend(times)

    # TODO: Include percentage of total and calculated total (not just reported total)
    for parent, times in sorted(n_parents.items(), key=lambda x: -np.array(x[1]).sum()):
        # Remove Overhead.
        times = [max(np.timedelta64(0, 'ns'), x - np.timedelta64(80, 'ns')) for x in times]
        times = np.array(times)

        print(f"{parent}", end=" ")

        # We can also work out statistics here:
        # Number of calls, mean call time, median, mode, outliers, etc.
        sum_ = str(times.sum())
        mean = times.mean()
        print(f"sum: {sum_} count: {len(times)}".replace(" nanoseconds", "ns"), end=" ")
        print(
            f"mean: {mean} median: {np.median(times)}".replace(" nanoseconds", "ns"),
            end="\n",
        )
