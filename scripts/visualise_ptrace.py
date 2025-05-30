from __future__ import annotations
from collections import defaultdict


# Why do we even need to reconstruct the call stack?
# 1) In order to be able to visualise the call stack
# 2) For accurate timing information. Consider the case where we
#   YieldFrom, AdvanceInputIterator, YieldFrom, YieldInto. Without 
#   tracking parent information we don't know what time goes into the
#   YieldIterator.


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

    def __str__(self) -> str:
        return f"({self.id} [{', '.join(x.id for x in self._children)}])"

    def __repr__(self) -> str:
        return str(self)

class Tree:
    def __init__(self):
        self._root_node: Node | None = None
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

    def __getitem__(self, node: Node):
        return self._nodes[node.id]

    def add_stack(self, operations, parent: Node | None) -> Node:
        assert len(operations) >= 1

        if parent is None:
            parent = Node(operations[-1][0])
            self._root_node = parent
            self._nodes[parent.id] = parent
            operations = operations[:-1]
            top = parent
        else:
            top = None

        for op in reversed(operations):
            opid = op[0]
            node = Node(opid)
            self._nodes[node.id] = node
            parent.add_child(node)
            parent = node

            if top is None:
                top = node

        # Returns the highest child on the stack, or the parent
        # if parent is None.
        return top

    def children(self, node: Node) -> list[Node]:
        return self._nodes[node.id]._children

    def parents(self, needle: Node) -> list[Node]:
        parents = []
        for (nid, node) in self._nodes.items():
            if needle in node._children:
                parents.append(node)
        return parents

    def get_node(self, node_id):
        return self._nodes[node_id]

    def __str__(self) -> str:
        ret = []
        queue = [self._root_node]
        while queue:
            node = queue.pop()
            ret.append(node)
            queue.extend(node._children)

        return str(ret)


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
    def active_node(self) -> Node:
        if self._cursor is None:
            raise KeyError("None is not a valid Node.")
        return self._cursor

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
                # FIXME: Hypothesis: The last element before a non-initial Call() must always
                # be a YieldFrom(ResolveNeighborsOuter). This experimentally seems to match
                # up. i.e. we assume that after we use Call() we implicitly add an inner
                # function call to a ResolveNeighborsInner().
                assert self._last_op is None or self._last_op[3].startswith("YieldFrom(ResolveNeighborsOuter")
                self._calling = True
                if self._last_op is not None:
                    self._call_list = [ self._last_op, op ]

                    # The cursor was incremented by one because of the last yield.
                    # However, as we pass the cursor down into the callstack, we
                    # actually want it to be one lower.

                    # If this assertion fails then the cursor is invalid, as it
                    # should only be None before initialisation.
                    assert self._cursor is not None

                    # If this assertion fails then for some reason the previous
                    # call had multiple (or zero) children. I guess it's because 
                    # the neighbors were not exhausted yet?
                    assert len(self._cursor._children) == 1
                    self._cursor = self._cursor._children[0]
                else:
                    self._call_list = [ op ]

            self._calls.append(opid)
        elif self._calling:
            # The cursor points to the parent before the stack started.
            # ... unless the parent was a yield (which is likely). In this case,
            # we abuse the yield above.
            # FIXME: Don't abuse the yield so much.
            self._cursor = self._callstack.add_stack( self._call_list, self._cursor )

            self._calling = False
            self._call_list = []

        if op_type.startswith("Call"):
            pass
        elif op_type.startswith("AdvanceInputIterator"):
            if parent not in self._calls:
                print("AdvanceInputIterator called with non-function parent.\n\t Operation: ", op)
            elif parent != self.active_node.id and parent != self._last_yield:
                print("AdvanceInputIterator called with incorrect function stack.", end="")
                self._print_debug_info(op)

            # If the last yield provided invalid results (such as not being of the right type)
            # the function may be called again.
            if parent != self.active_node.id:
                self.descend()
                if self.active_node.id != self._last_yield:
                    print("Active function does not match the last yielded", end="")
                    self._print_debug_info(op)
            elif len(self.active_node._children) != 1:
                # A function can randomly decide to stop dealing with its iterator
                # (e.g. if it's already found what it wants). In this case, it will
                # simply yield and move on to AdvanceInputIterator without
                # kindly letting us know what it has done. In order to resolve this,
                # we make two assumptions:
                # 1) That the lower opid child was called earlier
                # 2) The the function called earlier hasn't been exhausted
                # 3) The function called later has been finished, and that's 
                #   why we now move on to the next one.

                # We also kill the other child on the assumption that it is now finished.
                # If it's not, then it will probably cause hard to debug errors (joy!)
                assert len(self._cursor._children) == 2
                child_1 = int(self._cursor._children[0].id[5:-1])
                child_2 = int(self._cursor._children[1].id[5:-1])
                assert child_1 != child_2
                if child_1 < child_2:
                    self._cursor._children = self._cursor._children[:1]
                else:
                    # child_2 < child_1
                    self._cursor._children = self._cursor._children[1:]
                self.descend()
            else:        
                # We are not currently evaluating the function but it hasn't been exhausted yet.
                self.descend()
        elif op_type.startswith("YieldFrom"):
            if op_type == "YieldFrom(ResolveStartingVertices)":
                self.ascend()
            elif op_type == "YieldFrom(ResolveNeighborsOuter)":
                # ResolveNeighborsOuter creates its own iterator and thus
                # deserves to go on the stack.
                self._neighbors_stack.append(opid)
                self.ascend()
                # self._cursor = self._callstack.add_stack( [op], self._cursor )
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

                # Special-case handling because of the way ResolveNeighborsOuter inserts itself.
                # We need to double step out.
                # grandparent = self._ops[parent][0]
                # print(op, self._last_yield, grandparent)
                # if self._last_yield == grandparent:
                #     self.ascend()
                #     self.ascend()

                # We currently only correctly track ResolveNeighborsInner for
                # Call situations. Regardless, if we have the correct parent
                # we should ascend.
                if self._cursor.id == parent:
                    self.ascend()
                
                # FIXME: We really shouldn't ignore the general case, but I 
                # just can't see how to include it right now.
                # self.ascend()
            else:
                self.ascend()
            self._last_yield = parent
        elif op_type.startswith("OutputIteratorExhausted"):
            if parent != self.active_node.id and parent not in self._neighbors_stack:
                print("OutputIteratorExhausted called with incorrect function stack.", end="")
                self._print_debug_info(op)
            
            if parent not in self._neighbors_stack:
                self.ascend()
        elif op_type.startswith("YieldInto"):
            pass
        elif op_type.startswith("InputIteratorExhausted"):
            if parent != self.active_node.id:
                print("InputIteratorExhausted called with incorrect function stack.", end="")
                self._print_debug_info(op)
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
        print(f"{ind} Operation: {op}",f"{ind} Stack: {self._callstack}", f"{ind} Index: {self._cursor}", f"{ind} Last Yield: {self._last_yield}")

    def ascend(self):
        # Walk down the tree.
        parents = self._callstack.parents(self._cursor)
        if len(parents) != 1:
            print("Invalid number of ancestors when walking up the tree. If this is the final operation, you can safely ignore this error.")
            print("\t Children:", parents)
            print("\t Stack:", self._callstack)
            print("\t Cursor:", self._cursor)
        else:
            self._cursor = parents[0]

    def descend(self):
        # Walk down the tree.
        children = self._callstack.children(self._cursor)
        if len(children) != 1:
            print("Invalid number of descendants when walking down the tree.")
            print("\t Children:", children)
            print("\t Stack:", self._callstack)
            print("\t Cursor:", self._cursor)
        else:
            self._cursor = children[0]


player = TraceReplay()
for i, op in enumerate(groups[0], 1):
    op = (op[0], op[1], op[2], " ".join(op[3:]))
    debug = False
    if i in range(47, 53):
        debug = True
    player.progress(op, debug)