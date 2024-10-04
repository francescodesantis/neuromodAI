f = open("softhebb_env/pip_reqs.txt", "r")
dest = open("softhebb_env/pip_reqs_latest.txt", "w")
lines = f.readlines()
for line in lines:
    cut_index = line.find(">")
    if cut_index == -1:
        cut_index = line.find("=")
    dest.write(line[:cut_index]+"\n")
f.close()
dest.close()
