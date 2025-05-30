from dataclasses import dataclass
from collections import defaultdict

with open("enum_missing.ptrace.txt", encoding="utf-8") as fd:
    lines = fd.readlines()


# README:
# This is a backup copy of visualise that works for YieldInto but not
# YieldFrom and still has some warnings left.

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


@dataclass
class Op:
    id: Opid
    parent: Opid
    duration: str
    type: str



operations = []
for line in lines:
    opid, parent, duration, type = line.strip().split(" ", 3)
    opid = Opid(opid[5:-1])
    # Only Call may have invalid parents.
    assert not type.startswith("Call") or parent == "None"
    parent = Opid(None) if parent == "None" else Opid(parent[10:-2])
    operations.append(Op(opid, parent, duration, type))

def seek(start: int) -> (bool, int):
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
            return False
        elif op.type.startswith("Call"):
            if advances > 0:
                return False
            calls += 1
        elif op.type.startswith("AdvanceInputIterator"):
            advances += 1
        else:
            return False

# We can use parent data to reconstruct overall function call
# hierarchies. For now, we want to be able to map which 
# YieldIntos belong to each YieldFrom 
last_yield_from = None
last_yield_into = None

yield_intos = defaultdict(list)
advance_iterators = defaultdict(list)
advance_parents = set()
yielded_parents = set()
try: 
    i = 0
    while i < len(operations):
        op = operations[i]
        if op.type == "YieldInto":
            # No adjacent YieldIntos
            assert operations[i-i].type != "YieldInto" and operations[i+1].type != "YieldInto"

            # YieldInto is always followed by a YieldFrom
            assert operations[i+1].type.startswith("YieldFrom")

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
                assert operations[i+1].type == "YieldInto"
                yield_intos[operations[i+1].id].append(op.id)

            # Conversely, YieldFrom(ResolveNeighborsOuter) must always be followed by any 
            # number of Calls and an equal number of AdvanceInputIterators, and then a 
            # YieldFrom(ResolveNeighborsInner) or an OutputIteratorExhausted. 
            # (.e. YF C C A A YF is ok, but YF C A C YF is not).
            if op.type == "YieldFrom(ResolveNeighborsOuter)":
                valid, idx = seek(i+1)
                assert valid

                # Because i is always followed by a YieldFrom(ResolveNeighborsInner)
                if operations[idx].type == "YieldFrom(ResolveNeighborsInner)":
                    assert operations[idx+1].type == "YieldInto"
                    yield_intos[operations[idx+1].id].append(op.id)
                else:
                    # No clue what to do if it's an OutputIteratorExhausted.
                    print(f"Warning ({i}L): YieldFrom(ResolveNeighborsOuter) followed by OutputIteratorExhausted")
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
                    yielded_parents.discard(op.parent)
                else:
                    print(f"Warning ({i}L): Non-ResolveStartingVertices YieldFrom is not followed by YieldInto or an AdvanceInputIterator with the same id.")
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
                    print(f"Warning ({i}L): AdvanceInputIterator preceded by a YieldFrom with a different parent.")

        i += 1
except AssertionError as e:
    print(i, op)
    raise e

print(yield_intos)