import ray 
ray.init()
f = open("softhebb_env/pip_reqs.txt", "r")
dest = open("softhebb_env/pip_reqsX.txt", "w")
lines = f.readlines()
for line in lines:
    line = line.split("==")
    dest.write(line[0]+"\n")
f.close()
dest.close()