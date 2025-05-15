import orjson

DATA_PATH = R"C:\Users\josep\dev\gsoc\cargo\trustfall\scripts\example.ptrace.json"

with open(DATA_PATH, "rb") as fd:
	data = orjson.loads(fd.read())


class Operation:
	def __init__(self, opid: int, parent_opid, raw_data: dict):
		self.opid = opid
		self.parent_opid = parent_opid
		self.children = []

		self._data = raw_data
		self.time = None
		if self._data.get("time", None):
			secs, nanos = self._data["time"].values()
			self.time = f"{secs}s {nanos}ns"

	def __repr__(self) -> str:
		return f"Operation({self.opid}, {self.parent_opid}, {[x.opid for x in self.children]})"

# Create parent graph
roots = {}
operations = {}
for key in data["trace"]["ops"].values():
	parent = key["parent_opid"]
	if parent is None:
		op = Operation(key["opid"][0], parent, key)
		roots[op.opid] = op
	else:
		op = Operation(key["opid"][0], parent[0], key)
		operations[op.parent_opid].children.append(op)

	operations[op.opid] = op


def print_tree(roots: list[Operation], depth=0):
	if not roots:
		return

	for root in roots:
		print("-" * depth, root.opid, f"{root.time}")
		print_tree(root.children, depth+1)

print_tree(list(roots.values()))