def parse(filename):
    with open(filename,'r') as myfile:
        data = myfile.read()
    return data
def printToFile(stri,dst):
    with open(dst,"w") as myfile:
        myfile.write(stri)
