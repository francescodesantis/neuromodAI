import ray 
ray.init()
f = open("reqs.txt", "r")
dest = open("reqs_latest.txt", "w")
lines = f.readlines()
for line in lines:
    cut_index = line.find(">")
    dest.write(line[:cut_index]+"\n")
f.close()
dest.close()