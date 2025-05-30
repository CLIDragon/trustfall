from dataclasses import dataclass
from collections import defaultdict
from datetime import timedelta
from typing import ValuesView
import numpy as np
import math

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


class Duration:
    def __init__(self, value: str | list):
        """Converts a string of the form 3.7µs to a duration."""
        self._s = 0
        self._ms = 0
        self._micro_s = 0
        self._ns = 0

        if isinstance(value, list):
            if len(value) != 4:
                raise ValueError("Input list must have exactly 4 elements.")
            self._s, self._ms, self._micro_s, self._ns = value
        elif value.endswith("µs"):
            self._micro_s = float(value.removesuffix("µs"))
        elif value.endswith("ms"):
            self._ms = float(value.removesuffix("ms"))
        elif value.endswith("ns"):
            self._ns = float(value.removesuffix("ns"))
        elif value.endswith("s"):
            # This must be last, as it excludes the others.
            self._s = float(value.removesuffix("s"))
        elif not value:
            # Empty strings are just 0 durations. 
            # TODO: Should this change in future?
            pass
        else:
            raise ValueError(f"{value} is an invalid Duration.")

    def __sub__(self, other):
        if not isinstance(other, Duration):
            raise TypeError(f"Cannot subtract {other} from Duration.")

        new_s, rem = divmod(self._s - other._s, 1)
        new_ms, rem = divmod(self._ms - other._ms + rem * 1000, 1)
        new_micro_s, rem = divmod(self._micro_s - other._micro_s + rem * 1000, 1)
        new_ns = self._ns - other._ns + rem * 1000

        if abs(new_ns) > 1000:
            new_micro_s += new_ns // 1000
            new_ns = new_ns % 1000
        
        # TODO: Implement proper wrapping.
        return Duration([new_s, new_ms, new_micro_s, new_ns])

    def __add__(self, other):
        if not isinstance(other, Duration):
            raise TypeError(f"Cannot add {other} to Duration.")
        
        new_s, rem = divmod(self._s + other._s, 1)
        new_ms, rem = divmod(self._ms + other._ms + rem * 1000, 1)
        new_micro_s, rem = divmod(self._micro_s + other._micro_s + rem * 1000, 1)
        new_ns = self._ns + other._ns + rem * 1000

        if abs(new_ns) > 1000:
            new_micro_s += new_ns // 1000
            new_ns = new_ns % 1000
        
        # TODO: Implement proper wrapping.
        return Duration([new_s, new_ms, new_micro_s, new_ns])

    def __repr__(self) -> str:
        return f"Duration({self._s}s, {self._ms}ms, {self._micro_s}µs, {self._ns}ns)"

def duration_from(value: str) -> np.timedelta64:
    if value.endswith("µs"):
        rem, val = math.modf(float(value.removesuffix("µs")))
        return np.timedelta64(int(val), 'us') + duration_from(f"{rem*1000}ns")
    elif value.endswith("ms"):
        rem, val = math.modf(float(value.removesuffix("ms")))
        return np.timedelta64(int(val), 'ms') + duration_from(f"{rem*1000}µs")
    elif value.endswith("ns"):
        # FIXME: This is discarding values.
        rem, val = math.modf(float(value.removesuffix("ns")))
        return np.timedelta64(int(val), 'ns')
    elif value.endswith("s"):
        # This must be last, as it excludes the others.
        rem, val = math.modf(float(value.removesuffix("s")))
        return np.timedelta64(int(val), 's') + duration_from(f"{rem*1000}ms")
    elif not value:
        # Empty strings are just 0 durations. 
        # TODO: Should this change in future?
        return np.timedelta64('Nat')

@dataclass
class Op:
    id: Opid
    parent: Opid
    duration: np.timedelta64
    type: str


def seek(start: int, operations: list[Op]) -> (bool, int):
    """
    Look for the next YieldFrom(ResolveNeighborsInner) or OutputIteratorExhausted,
    counting calls and Advance InputIterators. Take start to be the first index
    to look from. 

    Matches patterns of the form:

    C{N}A{N}(YF|O)

    where C is a Call, A is an AdvanceInputIterator, YF is YieldFrom and O is
    OutputIteratorExhausted

    Requires:
        :i: to be a valid index of operations.

    Returns:
        True if the pattern matches, otherwise false.
        -1 if the pattern doesn't match, otherwise the index of the found element.
    """
    calls = 0
    advances = 0
    for i in range(start, len(operations)):
        op = operations[i]
        if op.type in ("YieldFrom(ResolveNeighborsInner)", "OutputIteratorExhausted"):
            if calls == advances:
                return (True, i)
            return (False, -1)
        elif op.type.startswith("Call"):
            if advances > 0:
                return (False, -1)
            calls += 1
        elif op.type.startswith("AdvanceInputIterator"):
            advances += 1
        else:
            return (False, -1)
    return (False, -1)

def calc_yields(filename):
    with open(filename, encoding="utf-8") as fd:
        lines = fd.readlines()

    repeat_warnings = False
    warned_output_iterator = False
    warned_starting_vertices = False
    warned_advance_input = False

    operations = []
    for line in lines:
        opid, parent, duration, type = line.strip().split(" ", 3)
        opid = Opid(opid[5:-1])
        # Only Call may have invalid parents.
        assert not type.startswith("Call") or parent == "None"
        parent = Opid(None) if parent == "None" else Opid(parent[10:-2])

        assert not type.startswith("YieldFrom") or duration != "None"
        assert not type.startswith("YieldInto") or duration != "None"
        duration = duration_from(duration[5:-1])
        operations.append(Op(opid, parent, duration, type))

    # YieldInto(id) : [YF, YF, ...]
    yield_intos = defaultdict(list)
    advance_iterators = defaultdict(list)
    advance_parents = set()
    yielded_parents = set()

    # YF(id) : [YI, YI, ...]
    yield_froms = defaultdict(list)
    try: 
        i = 0
        while i < len(operations):
            op = operations[i]
            if op.type == "YieldInto":
                # No adjacent YieldIntos
                assert operations[i-i].type != "YieldInto" and operations[i+1].type != "YieldInto"

                # YieldInto is always followed by a YieldFrom
                assert operations[i+1].type.startswith("YieldFrom")

                # YieldInto always has the same parent as the YieldFrom that follows it
                assert operations[i+1].parent == op.parent

                yield_froms[operations[i+1].id].append(op.id)

                if yielded_parents:
                    remove = set()
                    remove_keys = []
                    for parent in yielded_parents:
                        if op.parent != parent:
                            # print(op.parent, parent)
                            # print(f"Line {i}: {yielded_parents}")
                            remove.add(parent)

                            # TODO: Group these by parent, since that's all that matters
                            # right now.
                            for (opid, op_parent) in advance_iterators:
                                if op_parent == parent:
                                    yield_intos[op.id].extend(advance_iterators[(opid, op_parent)])
                                    remove_keys.append((opid, op_parent))

                            for key in remove_keys:
                                del advance_iterators[key]

                    yielded_parents = yielded_parents.difference(remove)
                    advance_parents = advance_parents.difference(remove)

            elif op.type.startswith("YieldFrom"):
                if op.parent in advance_parents:
                    yielded_parents.add(op.parent)

                if op.type == "YieldFrom(ResolveStartingVertices)":
                    # YieldFrom(ResolveStartingVertices) must be followed by a YieldInto
                    assert operations[i+1].type == "YieldInto"
                    yield_intos[operations[i+1].id].append(op.id)
                elif op.type != "YieldFrom(ResolveNeighborsInner)":
                    # YieldFrom is always preceded by a YieldInto, unless it is a YieldFrom(ResolveStartingVertices)
                    # or a YieldFrom(ResolveNeighborsInner)
                    assert operations[i-1].type == "YieldInto"

                if op.type == "YieldFrom(ResolveNeighborsInner)":
                    assert operations[i+1].type in ("YieldInto", "OutputIteratorExhausted")
                    if operations[i+1].type == "YieldInto":
                        yield_intos[operations[i+1].id].append(op.id)
                    else:
                        assert operations[i+2].type == "YieldInto"
                        yield_intos[operations[i+2].id].append(op.id)

                # Conversely, YieldFrom(ResolveNeighborsOuter) must always be followed by any 
                # number of Calls and an equal number of AdvanceInputIterators, and then a 
                # YieldFrom(ResolveNeighborsInner) or an OutputIteratorExhausted. 
                # (.e. YF C C A A YF is ok, but YF C A C YF is not).
                if op.type == "YieldFrom(ResolveNeighborsOuter)":
                    valid, idx = seek(i+1, operations)
                    assert valid

                    # Because i is always followed by a YieldFrom(ResolveNeighborsInner)
                    if operations[idx].type == "YieldFrom(ResolveNeighborsInner)":
                        assert operations[idx+1].type in ("YieldInto", "OutputIteratorExhausted")
                        if operations[idx+1].type == "YieldInto":
                            yield_intos[operations[idx+1].id].append(op.id)
                        else:
                            # The OutputIteratorExhausted, must be followed by a YieldInto with a different parent.
                            assert operations[idx+2].type == "YieldInto"
                            yield_intos[operations[idx+2].id].append(op.id)
                    else:
                        # Must be an OutputIteratorExhausted.

                        # The OutputIteratorExhausted must be followed by an AdvanceInputIterator
                        # which has the same parent. Then we belong to the AdvanceInputIterator
                        assert operations[idx].type == "OutputIteratorExhausted"
                        assert operations[idx+1].type in ("AdvanceInputIterator", "YieldInto")

                        if operations[idx+1].type == "AdvanceInputIterator":
                            assert operations[idx+1].parent == op.parent

                            # This cannot get assigned to an earlier child because seek() guarantees
                            # that the only middle entries are Call and AdvanceInputIterator.
                            advance_iterators[(operations[idx+1].id, op.parent)].append(op.id)
                            advance_parents.add(op.parent)

                            # The discard is necessary because we set it above.
                            yielded_parents.discard(op.parent)
                        else:
                            # Must be YieldInto
                            yield_intos[operations[idx+1].id].append(op.id)

                        # if repeat_warnings or not warned_output_iterator:
                        #     print(f"Warning ({i}L): YieldFrom(ResolveNeighborsOuter) followed by OutputIteratorExhausted")
                        #     warned_output_iterator = True
                else:
                    # 
                    assert not operations[i+1].type.startswith("YieldFrom")

                # A YieldFrom followed by a YieldInto must have different parents.
                assert operations[i+1].type != "YieldInto" or operations[i+1].parent != op.parent

                if op.type in ("YieldFrom(ResolveProperty)", "YieldFrom(ResolveCoercion)"):
                    if operations[i+1].type == "YieldInto":
                        yield_intos[operations[i+1].id].append(op.id)
                    elif operations[i+1].type == "AdvanceInputIterator" and operations[i+1].parent == op.parent:
                        advance_iterators[(operations[i+1].id, op.parent)].append(op.id)
                        advance_parents.add(op.parent)

                        # The discard is necessary because we set it above.
                        yielded_parents.discard(op.parent)
                    else:
                        if repeat_warnings or not warned_starting_vertices:
                            print(f"Warning ({i}L): Non-ResolveStartingVertices YieldFrom is not followed by YieldInto or an AdvanceInputIterator with the same id.")
                            warned_starting_vertices = True
            elif op.type.startswith("AdvanceInputIterator"):
                if op.parent in advance_parents:
                    yielded_parents.discard(op.parent)
                # If AdvanceInputIterator is preceded by YieldFrom, then the parents of the
                # two must be the same.
                if operations[i-1].type.startswith("YieldFrom"):
                    if operations[i-1].parent == op.parent:
                        # Then the following operations belong to the AdvanceInputIterator 
                        # and the AdvanceInputIterator belongs to the next YieldFrom with a 
                        # different parent that comes after a YieldFrom of the same parent. 
                        pass
                    else:
                        if repeat_warnings or not warned_advance_input:
                            print(f"Warning ({i}L): AdvanceInputIterator preceded by a YieldFrom with a different parent.")
                            warned_advance_input = True

            i += 1
    except AssertionError as e:
        print(i, op)
        # raise e

    return yield_intos, yield_froms, operations



import os


for entry in os.scandir(R"C:\Users\josep\dev\gsoc\cargo\trustfall\scripts\outputs"):
    if not entry.is_file():
        continue
    # if entry.name not in ("method_parameter_count_changed.ron.ptrace.txt",):
    #     continue
    # if entry.name < "method_parameter_count_changed.ron.ptrace.txt":
    #     continue

    print(f"Query {entry.name.split('.')[0]}")
    yield_intos, yield_froms, operations = calc_yields(entry.path)

#     # Work out how much time was spent in each function call (modeled by Parent)
#     parents = defaultdict(list)
#     for yf, yis in yield_froms.items():
#         assert len(yis) == 1

#         yi = yis[0]
#         op = operations[int(yf)-1]

#         assert op.parent != Opid(None)
#         parents[op.parent].append(op.duration - operations[int(yi)-1].duration)

#     for parent, times in parents.items():
#         print(f"{operations[int(parent)-1].type[5:-1]} takes {sum(times, np.timedelta64(0))}")

#         # We can also work out statistics here:
#         # Number of calls, mean call time, median, mode, outliers, etc.