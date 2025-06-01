with open("out.txt") as fd:
	lines = fd.readlines()
data = []
for i in range(2, len(lines), 3):
	if lines[i].startswith("Total"):
		data.append(lines[i])
	else:
		time_str = lines[i+1].strip().removeprefix("Time: ").removesuffix("µs")
		if time_str.endswith("ms"):
			time_str = str(float(time_str.removesuffix("ms")) * 1000)
		else:
			time_str = time_str.removesuffix("Â")
		data.append(lines[i].strip().removeprefix("Query ") + "\t" + time_str + "\t" + lines[i+2].strip().removeprefix("Operations: "))

with open("out-ed.txt", "w") as fd:
	fd.write("\n".join(data))