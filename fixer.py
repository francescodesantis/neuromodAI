f = open("requirements.txt", "r")
dest = open("reqs.txt", "w")
lines = f.readlines()
for line in lines:
    line = line.replace("=", ">", 1)
    dest.write(line)
f.close()
dest.close()