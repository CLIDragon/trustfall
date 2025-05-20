from __future__ import annotations
from collections import defaultdict


with open("enum_missing.ptrace.txt") as fd:
    data = fd.read().strip().split("\n")

linear = data
print(len(linear), "operations")

# Preprocess data.
groups = []
group = []
for line in linear:
    if line == "":
        groups.append(group)
        group = []
    else:
        group.append(line)
groups.append(group)

for group in groups:
    for i in range(len(group)):
        line = group[i]
    
        parts = line.split(" ")
        for j in range(len(parts)):
            part = parts[j]
            if part.startswith("Some("):
                parts[j] = part.removeprefix("Some(").removesuffix(")")
        
        group[i] = tuple(parts)


class Node:
    def __init__(self, opid):
        self.id = opid
        self._children = []

    def add_child(self, node: Node):
        self._children.append(node)

class Tree:
    def __init__(self):
        self._root_node = None
        self._nodes = {}

    def add_node(self, node: Node, parent: Node | None):
        # Only valid to set parent to none when the root is None.
        # Assumes:
        #   The parent is a node in the tree.
        if parent is None and self._root_node is not None:
            raise ValueError("parent is None when root node exists.")
        elif parent is None:
            assert not self._nodes

            self._root_node = node
            self._nodes[node.id] = node
        else:
            self._nodes[parent.id].add_child(node)
            self._nodes[node.id] = node

    def __getitem__(self, node_id):
        return self._nodes[node_id]

    def add_stack(self, operations, parent: Node | None):
        if parent is None:
            parent = Node(operations[-1][0])
            self._root_node = parent.id
            self._nodes[parent.id] = parent
            operations = operations[:-1]

        for op in reversed(operations):
            opid = op[0]
            node = Node(opid)
            self._nodes[node.id] = node
            parent.add_child(node)
            parent = node


class TraceReplay:
    def __init__(self):
        self._calls = []
        self._ops = {}
        self._hierarchy = defaultdict(list)
        self._roots = []

        self._callstack = Tree()
        self._neighbors_stack = []
        self._cursor: Node | None = None
        self._last_yield = None

        self._last_op = None

        # True when we are collecting a list of function calls.
        self._calling = True
        self._call_list = []

    @property
    def active_function(self) -> Node:
        return self._callstack[self._cursor]

    def progress(self, op, debug=False):
        """Progresses the internal state according to the operation."""
        (opid, parent, time, op_type) = op
        self._ops[opid] = (parent, time, op_type)

        if debug:
            self._print_debug_info(op, False)
        
        if parent is not None:
            # parent must always exist otherwise there is an error.
            self._hierarchy[parent].append(opid)
        else:
            self._roots.append(opid)

        if op_type.startswith("Call"):
            # FIXME: Currently assumes that function calls are always evaluated
            # on top of the stack from wherever they are until exhausted.
            # FIXME: Assumption: We don't leave the stack unless we're about to be exhausted.
            if self._calling:
                self._call_list.append(op)
            else:
                self._calling = True
                self._call_list = [ op ]
        else:
            # The cursor points to the parent before the stack started.
            self._callstack.add_stack( self._call_list, self._cursor )
            self._cursor = self._call_list[-1][0]
            self._calling = False
            self._call_list = []

        if op_type.startswith("AdvanceInputIterator"):
            if parent not in self._calls:
                print("AdvanceInputIterator called with non-function parent.\n\t Operation: ", op)
            elif parent != self.active_function and parent != self._last_yield:
                print("AdvanceInputIterator called with incorrect function stack.", end="")
                self._print_debug_info(op)

            # If the last yield provided invalid results (such as not being of the right type)
            # the function may be called again.
            if parent != self.active_function:
                self.descend()
                if self.active_function != self._last_yield:
                    print("Active function does not match the last yielded", end="")
                    self._print_debug_info(op)
            else:        
                # We are not currently evaluating the function but it hasn't been exhausted yet.
                self.descend()
        elif op_type.startswith("YieldFrom"):
            self._last_yield = parent
            if op_type == "YieldFrom(ResolveStartingVertices)":
                self.ascend()
            elif op_type == "YieldFrom(ResolveNeighborsOuter)":
                # ResolveNeighborsOuter creates its own iterator and thus
                # deserves to go on the stack. It should go one behind so
                # that the yield from ResolveNeighborsInner moves to the
                # parent properly.
                # self._callstack.insert(self._index, opid)

                # It goes on its own stack in order to avoid numerical
                # conflicts.
                # self._neighbors_stack.append(opid)

                self._tag += 1
                self._callstack.insert(self._cursor + 1, (opid, self._tag))
                self.ascend()
            elif op_type == "YieldFrom(ResolveNeighborsInner)":
                if parent not in self._neighbors_stack:
                    print("Yielding from ResolveNeighborsInner without a valid parent.\n\tOperation:", op)
                # TODO: Is it possible to have multiple subsequent ResolveNeighborsInner?
                # Currently assume it is not possible.
                # Move the index if we are a ResolveNeighbors function. Otherwise the yield is actually
                # from an internal function, similar to from ResolveStartingNeighbors.
                # TODO: Bookkeep properly.

                # If the parent of the previous function call was a ResolveNeighbors function (or,
                # equivalently, the last operation was a YieldFrom(ResolveNeighborsOuter)) then 
                # this is changing the active function. Otherwise, it's simply doing the equivalent
                # of a AdvanceInputIterator followed by a YieldFrom without the AdvanceInputIterator
                # (YieldFrom(ResolveNeighborsOuter) has an implicit Call under this model).
                # prev_parent = self._ops[self._last_op[0]][0]
                # if "ResolveNeighbors" in self._ops[prev_parent][2]:
                
                # Special-case handling because of the way ResolveNeighborsOuter inserts itself.
                self.ascend()
            else:
                self.ascend()
        elif op_type.startswith("OutputIteratorExhausted"):
            # if parent != self.active_function:
            #     print("OutputIteratorExhausted called with incorrect function stack.", end="")
            #     self._print_debug_info(op)
            pass
        elif op_type.startswith("YieldInto"):
            pass
        elif op_type.startswith("InputIteratorExhausted"):
            # if parent != self.active_function:
            #     print("InputIteratorExhausted called with incorrect function stack.", end="")
            #     self._print_debug_info(op)
            pass

        elif op_type.startswith("ProduceQueryResult"):
            pass
        else:
            print("Unknown operation type:", op_type)

        self._last_op = op
    
    def _print_debug_info(self, op, indent=True):
        if indent:
            ind = "\n\t"
        else:
            ind = "\n"
        print(f"{ind} Operation: {op}",f"{ind} Stack: {self._callstack}", f"{ind} Index: {self._cursor} ({self.active_function})", f"{ind} Last Yield: {self._last_yield}")

    def ascend(self):
        cur = self._callstack[self._cursor]
        fut = self._callstack[self._cursor + 1]
        # If the tags are the same then we're good to go.
        # We can always move right to a greater tag.
        if fut[1] >= cur[1]:
            self._cursor += 1
            return

        # Otherwise look left for the first tag that's one less
        # than this one. This is because when we embed a new stack
        # its parent is the inverse of what it started next to.
        for i in range(self._cursor - 1, -1, -1):
            if self._callstack[i][1] == cur[1] - 1:
                self._cursor = i
                break
        else:
            print("Could not find a matching pair with a lower tag when incrementing.")
            print("\t Stack:", self._callstack)
            print("\t Index:", self._cursor)

    def descend(self):
        cur = self._callstack[self._cursor]
        fut = self._callstack[self._cursor - 1]
        # If the tags are the same then we're good to go.
        # We can always move left to a smaller tag.
        # FIXME: Should this only be one less?
        if fut[1] <= cur[1]:
            self._cursor -= 1
            return

        # Otherwise look left for the first tag that's one less
        # than this one. This is because when we embed a new stack
        # its parent is the inverse of what it started next to.
        for i in range(self._cursor - 1, -1, -1):
            if self._callstack[i][1] == cur[1] - 1:
                self._cursor = i
                break
        else:
            print("Could not find a matching pair with a lower tag when decrementing.")
            print("\t Stack:", self._callstack)


player = TraceReplay()
for i, op in enumerate(groups[0], 1):
    op = (op[0], op[1], op[2], " ".join(op[3:]))
    debug = False
    if i in range(29, 46):
        debug = True
    player.progress(op, debug)