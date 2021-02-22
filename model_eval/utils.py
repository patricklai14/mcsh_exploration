import pathlib

#mem is in terms of GB
def create_pbs(location, job_name, command, mem=2, time="00:30:00"):
    pbs_filename = pathlib.Path(location) / "{}.pbs".format(job_name)
    pbs_file = open(pbs_filename, "w+")
    
    pbs_file.write("#PBS -N {}\n".format(job_name))
    pbs_file.write("#PBS -l nodes=1:ppn=1\n")
    pbs_file.write("#PBS -l pmem={}gb\n".format(mem))
    pbs_file.write("#PBS -l walltime={}\n".format(time))
    pbs_file.write("#PBS -q pace-ice\n")
    pbs_file.write("#PBS -j oe\n")
    pbs_file.write("#PBS -o {}\n".format(pathlib.Path(location) / "{}.out".format(job_name)))
    pbs_file.write("\n")

    pbs_file.write("module load anaconda3/2019.10\n")
    pbs_file.write("conda activate amptorch\n")
    pbs_file.write(command)
    pbs_file.write("\n")

    pbs_file.close()

    return pbs_filename
