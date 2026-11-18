def filepath(severname):
    if severname == "Trillium":
        location = "/home/yuzhengx/RMFtool/Restricted_Gutzwiller"
        outlocation = "/scratch/yuzhengx/RMFtool/Restricted_Gutzwiller/Output/"
        return location, outlocation
    elif severname == "TrilliumGPU":
        location = "/home/yuzhengx/RMFtool/Restricted_Gutzwiller"
        outlocation = "/scratch/yuzhengx/RMFtool/Restricted_Gutzwiller/Output/"
        return location, outlocation
    elif severname =="Local":
        location = "./"
        outlocation = "./Output/"
        return location, outlocation
    else:
        raise Exception("Severname is not in Filepath.py.")