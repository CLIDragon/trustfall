from dataclasses import dataclass
from collections import defaultdict
from datetime import timedelta
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

    # YieldInto(id) : [YF, YF, ...]
    # Also: ProduceQueryResult(id): [YF, YF, ...]
    yield_intos = defaultdict(list)

    # parent: [YF, YF, ...]
    advance_iterators = defaultdict(list)
    advance_parents = set()
    yielded_parents = set()

    # YF(id) : YI or None (if RSV, RNI)
    yield_froms = {}
    try:
        i = 0
        while i < len(operations):
            op = operations[i]
            if op.type == "YieldInto":
                # No adjacent YieldIntos
                assert (
                    operations[i - i].type != "YieldInto"
                    and operations[i + 1].type != "YieldInto"
                )

                # YieldInto is always followed by a YieldFrom
                assert operations[i + 1].type.startswith("YieldFrom")

                # YieldInto always has the same parent as the YieldFrom that follows it
                assert operations[i + 1].parent == op.parent

                assert op.id not in yield_froms
                yield_froms[operations[i + 1].id] = op.id

                if yielded_parents:
                    remove = set()
                    remove_keys = []

                    # Hypothesis: yielded_parents can never be larger than 1.
                    for parent in yielded_parents:
                        if op.parent != parent:
                            # print(op.parent, parent)
                            # print(f"Line {i}: {yielded_parents}")
                            remove.add(parent)

                            yield_intos[op.id].extend(advance_iterators[parent])
                            remove_keys.append(parent)

                    for key in remove_keys:
                        del advance_iterators[key]

                    yielded_parents = yielded_parents.difference(remove)
                    advance_parents = advance_parents.difference(remove)

            elif op.type.startswith("YieldFrom"):
                if op.parent in advance_parents:
                    yielded_parents.add(op.parent)

                if op.type == "YieldFrom(ResolveStartingVertices)":
                    # YieldFrom(ResolveStartingVertices) must be followed by a YieldInto
                    assert operations[i + 1].type in ("YieldInto", "ProduceQueryResult")
                    yield_intos[operations[i + 1].id].append(op.id)

                    # ResolveStartingVertices does not have a YieldInto.
                    assert op.id not in yield_froms
                    yield_froms[op.id] = None
                elif op.type != "YieldFrom(ResolveNeighborsInner)":
                    # YieldFrom is always preceded by a YieldInto, unless it is a YieldFrom(ResolveStartingVertices)
                    # or a YieldFrom(ResolveNeighborsInner)
                    assert operations[i - 1].type == "YieldInto"

                if op.type == "YieldFrom(ResolveNeighborsInner)":
                    # This is treated exactly the same as in the case beneath
                    # YieldFrom(ResolveNeighborsOuter). Generally this is basically
                    # duplicate code.
                    # FIXME: De-duplicate this code.

                    # ResolveNeighborsInner does not have a YieldInto.
                    assert op.id not in yield_froms
                    yield_froms[op.id] = None

                    idx = i
                    # There may be any number of following ResolveNeighborsInner,
                    # but in the end they will all have the same YieldInto, it's
                    # just a question of how to get them all there.
                    if operations[idx + 1].type == "YieldFrom(ResolveNeighborsInner)":
                        for j in range(idx + 2, len(operations)):
                            if operations[j].type != "YieldFrom(ResolveNeighborsInner)":
                                idx = j - 1
                                break
                            else:
                                assert operations[j].parent == op.parent

                    assert operations[idx + 1].type in (
                        "YieldInto",
                        "OutputIteratorExhausted",
                        "AdvanceInputIterator",
                        "ProduceQueryResult",
                    )
                    if operations[idx + 1].type == "YieldInto":
                        yield_intos[operations[idx + 1].id].append(op.id)
                    elif operations[idx + 1].type == "OutputIteratorExhausted":
                        # The OutputIteratorExhausted, must be followed by a YieldInto with a different parent.
                        assert operations[idx + 2].type in (
                            "YieldInto",
                            "AdvanceInputIterator",
                        )
                        if operations[idx + 2].type == "YieldInto":
                            assert operations[idx + 2].parent != op.parent
                            yield_intos[operations[idx + 2].id].append(op.id)
                        else:
                            # See YieldFrom(ResolveNeighborsInner) for details.
                            yielded_parents.discard(op.parent)

                            advance_parents.add(op.parent)
                            advance_iterators[op.parent].append(op.id)
                    elif operations[idx + 1].type == "AdvanceInputIterator":
                        # AdvanceInputIterator. We treat this exactly the same
                        # as if we were any other YieldFrom encountering a
                        # AdvanceInputIterator. See the YieldFrom(ResolveProperty)
                        # case below for an explanation.
                        advance_iterators[op.parent].append(op.id)
                        advance_parents.add(op.parent)

                        # The discard is necessary because we set it above.
                        yielded_parents.discard(op.parent)
                    elif operations[idx + 1].type == "ProduceQueryResult":
                        yield_intos[operations[idx + 1].id].append(op.id)

                # Conversely, YieldFrom(ResolveNeighborsOuter) must always be followed by any
                # number of Calls and an equal number of AdvanceInputIterators, and then a
                # YieldFrom(ResolveNeighborsInner) or an OutputIteratorExhausted.
                # (.e. YF C C A A YF is ok, but YF C A C YF is not).
                if op.type == "YieldFrom(ResolveNeighborsOuter)":
                    valid, idx = seek(i + 1, operations)
                    assert valid

                    if idx != i + 1:
                        # If there is a C A set after this yield, then we clearly
                        # shouldn't be adding to the next YieldInto, as it will
                        # be from a much lower nested layer.
                        yielded_parents.discard(op.parent)

                        advance_parents.add(op.parent)
                        advance_iterators[op.parent].append(op.id)
                    # Because i is always followed by a YieldFrom(ResolveNeighborsInner)
                    # or an OutputIteratorExhausted or a AdvanceInputIterator.
                    elif operations[idx].type == "YieldFrom(ResolveNeighborsInner)":
                        # There may be any number of following ResolveNeighborsInner,
                        # but in the end they will all have the same YieldInto, it's
                        # just a question of how to get them all there.
                        if (
                            operations[idx + 1].type
                            == "YieldFrom(ResolveNeighborsInner)"
                        ):
                            for j in range(idx + 2, len(operations)):
                                if (
                                    operations[j].type
                                    != "YieldFrom(ResolveNeighborsInner)"
                                ):
                                    idx = j - 1
                                    break
                                else:
                                    assert operations[j].parent == op.id
                            else:
                                # Every remaining operation must be a YieldFrom(ResolveNeighborsInner)
                                assert False

                        # TODO: Test that ProduceQueryResult for ResolveNeighbors works.
                        # This may be difficult to find, because I am unsure any non-trivial
                        # query will end with a ResolveNeighbors.
                        assert operations[idx + 1].type in (
                            "YieldInto",
                            "OutputIteratorExhausted",
                            "AdvanceInputIterator",
                            "ProduceQueryResult",
                        )

                        if operations[idx + 1].type == "YieldInto":
                            yield_intos[operations[idx + 1].id].append(op.id)
                        elif operations[idx + 1].type == "OutputIteratorExhausted":
                            # The OutputIteratorExhausted, must be followed by a YieldInto
                            # with a different parent or an AdvanceInputIterator
                            assert operations[idx + 2].type in (
                                "YieldInto",
                                "AdvanceInputIterator",
                            )
                            if operations[idx + 2].type == "YieldInto":
                                assert operations[idx + 2].parent != op.parent
                                yield_intos[operations[idx + 2].id].append(op.id)
                            else:
                                # See YieldFrom(ResolveNeighborsInner) for details.
                                yielded_parents.discard(op.parent)

                                advance_parents.add(op.parent)
                                advance_iterators[op.parent].append(op.id)
                        elif operations[idx + 1].type == "AdvanceInputIterator":
                            # AdvanceInputIterator. We treat this exactly the same
                            # as if we were any other YieldFrom encountering a
                            # AdvanceInputIterator. See the YieldFrom(ResolveProperty)
                            # case below for an explanation.
                            advance_iterators[op.parent].append(op.id)
                            advance_parents.add(op.parent)

                            # The discard is necessary because we set it above.
                            yielded_parents.discard(op.parent)
                        elif operations[idx + 1].type == "ProduceQueryResult":
                            yield_intos[operations[idx + 1].id].append(op.id)

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
                            advance_iterators[op.parent].append(op.id)
                            advance_parents.add(op.parent)

                            # The discard is necessary because we set it above.
                            yielded_parents.discard(op.parent)
                        else:
                            # Must be YieldInto
                            yield_intos[operations[idx + 1].id].append(op.id)

                        # if repeat_warnings or not warned_output_iterator:
                        #     print(f"Warning ({i}L): YieldFrom(ResolveNeighborsOuter) followed by OutputIteratorExhausted")
                        #     warned_output_iterator = True
                elif op.type != "YieldFrom(ResolveNeighborsInner)":
                    #
                    assert not operations[i + 1].type.startswith("YieldFrom")

                # A YieldFrom followed by a YieldInto must have different parents.
                assert (
                    operations[i + 1].type != "YieldInto"
                    or operations[i + 1].parent != op.parent
                )

                if op.type in (
                    "YieldFrom(ResolveProperty)",
                    "YieldFrom(ResolveCoercion)",
                ):
                    if operations[i + 1].type == "YieldInto":
                        yield_intos[operations[i + 1].id].append(op.id)
                    elif operations[i + 1].type == "AdvanceInputIterator":
                        # If the parent is the same, then we are simply repeating
                        # a loop. If the parent is different, then ...
                        # ASSUME if the parent is different, then we were just part
                        # of a big stack following a CCAA call stack.
                        # TODO: Test this assumption.
                        advance_iterators[op.parent].append(op.id)
                        advance_parents.add(op.parent)

                        # The discard is necessary because we set it above.
                        yielded_parents.discard(op.parent)
                    elif operations[i + 1].type.startswith("ProduceQueryResult"):
                        yield_intos[operations[i + 1].id].append(op.id)
                    else:
                        if repeat_warnings or not warned_starting_vertices:
                            print(
                                f"Warning ({i + 1}L): Non-ResolveStartingVertices YieldFrom is not followed by YieldInto or an AdvanceInputIterator."
                            )
                            warned_starting_vertices = True
            elif op.type.startswith("AdvanceInputIterator"):
                if op.parent in advance_parents:
                    yielded_parents.discard(op.parent)
                # If AdvanceInputIterator is preceded by YieldFrom, then the parents of the
                # two must be the same.
                if operations[i - 1].type.startswith("YieldFrom"):
                    if operations[i - 1].parent == op.parent:
                        # Then the following operations belong to the AdvanceInputIterator
                        # and the AdvanceInputIterator belongs to the next YieldFrom with a
                        # different parent that comes after a YieldFrom of the same parent.
                        pass
                    else:
                        # If the preceding yield has a different parent, then assume that
                        # this is the end of a CCAA call stack and move on.

                        # if repeat_warnings or not warned_advance_input:
                        #     print(f"Warning ({i+1}L): AdvanceInputIterator preceded by a YieldFrom with a different parent.")
                        #     warned_advance_input = True
                        pass

            i += 1
    except AssertionError as e:
        print(i + 1, op)
        raise e

    # print("Advance Parents", advance_parents)

    return yield_intos, yield_froms, advance_iterators, operations, total_time


import os


def format_time(t: np.timedelta64) -> str:
    num, unit = str(t).split(" ")
    if unit == "nanoseconds":
        return f"{num}ns"
    elif unit == "microseconds":
        return f"{num}µs"
    return f"{num} {unit}"


np.set_printoptions(formatter={"timedelta": format_time})


ALLOW_LIST = [
    "auto_trait_impl_removed",
    "derive_trait_impl_removed",
    "sized_impl_removed",
    "trait_method_added",
    "trait_method_default_impl_removed",
    "trait_method_missing",
    "trait_method_now_doc_hidden",
    "trait_method_parameter_count_changed",
    "trait_method_requires_different_const_generic_params",
    "trait_method_requires_different_generic_type_params",
    "trait_method_unsafe_added",
    "sized_impl_removed",
    "trait_method_marked_deprecated",
]

# ALLOW_LIST = [
#     "feature_missing",
#     "feature_not_enabled_by_default",
#     "function_export_name_changed",
#     "exported_function_changed_abi"
# ]

ALLOW_LIST = [x + ".ron.ptrace.txt" for x in ALLOW_LIST]
import re

for entry in os.scandir(
    R"C:\Users\josep\dev\gsoc\cargo\trustfall\scripts\outputs"
):
    if not entry.is_file():
        continue
    if entry.name not in ALLOW_LIST:
        continue
    # if entry.name not in ("trait_method_added.ron.ptrace.txt",):
    #     continue
    # if entry.name < "method_parameter_count_changed.ron.ptrace.txt":
    #     continue

    print(f"=== {entry.name.split('.')[0]} === ")
    yield_intos, yield_froms, advance_iterators, operations, total_time = calc_yields(
        entry.path
    )
    print(f"Time: {total_time}")

    overhead_offset = 100  # ns
    ignore_overhead = False
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
            resolve_neighbors[operations[int(parent.id) - 1].parent][parent.id].append(
                time
            )
        else:
            parents[parent.id].append(time)

    # print(resolve_neighbors)

    # Collapse parents with the same name.
    n_parents = defaultdict(list)
    for parent, times in parents.items():
        par = operations[int(parent) - 1]
        n_parents[re.sub(r"Vid\(.*?\), ", "", par.type[5:-1])].extend(times)

    for parent, times in sorted(n_parents.items(), key=lambda x: -np.array(x[1]).sum()):
        # Remove Overhead.
        # times = [max(np.timedelta64(0, 'ns'), x - np.timedelta64(80, 'ns')) for x in times]
        times = np.array(times)


        # if times.sum() < 5_000:
        #     # print()
        #     continue
        print(f"{parent}", end=" ")

        # We can also work out statistics here:
        # Number of calls, mean call time, median, mode, outliers, etc.
        sum_ = times.sum()
        mean = times.mean()
        print(f"sum: {sum_} count: {len(times)}", end=" ")
        # FIXME: This isn't actually the median.
        print(f"mean: {mean} median: {np.median(times)}", end="\n")

    continue

    for parent, times in parents.items():
        times = np.array(times)
        parent = operations[int(parent) - 1]

        print(f"{parent.id} {parent.type[5:-1]}", end=" ")

        # We can also work out statistics here:
        # Number of calls, mean call time, median, mode, outliers, etc.
        sum_ = times.sum()
        mean = times.mean()
        print(f"sum: {sum_} count: {len(times)}", end=" ")
        # FIXME: This isn't actually the median.
        print(f"mean: {mean} median: {np.median(times)}", end="\n")

        if not print_neighbors:
            continue

        for i, inner_times in resolve_neighbors.get(parent.id, {}).items():
            inner_times = np.array(inner_times)
            print(f"\t{i}: {inner_times.sum()} {len(inner_times)} {inner_times.mean()}")

    # Assuming that it is impossible for a function to YieldInto itself,
    # the total time spent in each function is given by the unadjusted sum of
    # all of its YieldFroms.

    # parents = defaultdict(list)
    # resolve_neighbors = defaultdict(lambda: defaultdict(list))
    # for yf, yi in yield_froms.items():
    #     time  = operations[int(yf)-1].duration
    #     op = operations[int(yf)-1]
    #     parent = operations[int(op.parent)-1]

    #     # Separate RNO
    #     if parent.type == "YieldFrom(ResolveNeighborsOuter)":
    #         resolve_neighbors[operations[int(parent.id)-1].parent][parent.id].append(time)
    #     else:
    #         parents[parent.id].append(time)

    # Test raw duration vs calc duration.
    for yi, yfs in yield_intos.items():
        # if yi not in (Opid("5200"), Opid("5002")):
        #     continue

        # print(sorted(yfs))

        duration = np.timedelta64(0)

        for yf in yfs:
            duration += operations[int(yf) - 1].duration

        raw_duration = operations[int(yi) - 1].duration
        if raw_duration < (duration - np.timedelta64(10, "ns")):
            # raw duration shouldn't be less than calc duration.
            print(sorted(yfs))
            print(
                f"Raw duration for {yi} less than calc duration: ({raw_duration} < {duration})"
            )
        elif raw_duration > (duration + np.timedelta64(1000, "ns")):
            # It's okay for raw to be greater than calc, as that just means time spent
            # in the operation.
            pass
            # print(sorted(yfs))
            # print(f"Raw duration for {yi} greater than calc duration: ({raw_duration} > {duration})")
            # print(raw_duration, duration)

    # break

    # for yfs in yield_intos.values():
    #     if Opid("433") in yfs:
    #         print(yfs)
