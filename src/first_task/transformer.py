def toRGB(mark):
    if (mark == '0'):
        return [0, 0, 0]
    elif (mark == '1'):
        return [1, 0.1, 0.1]
    elif (mark == '2'):
        return [0.1, 1, 0.1]
    else:
        return [0.1, 0.1, 1]

def transformInputFile(fileName):
    file = open(fileName, "r")
    lines = file.readlines()
    arr = []
    for line in lines:
        tmpArr = []
        for index in range(0, len(line) - 1):
            tmpArr.append(toRGB(line[index]))
        arr.append(tmpArr)
    return arr
