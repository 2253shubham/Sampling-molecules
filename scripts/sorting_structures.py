#  load append "file:///E:/phd work/python script analysis/sampling_configurations_MCCCS/pcracking_nc4/sampled_configurations/MFI-pcracking-18-strcuture.xyz"
#  load append "file:///E:/phd work/aldol condensation/protolytic_cracking/distinct_configurations/intersection/distinct_intersection_num_2_structure.xyz"
#  load append "file:///E:/phd work/aldol condensation/reactant_state_pcracking/distinct_configurations/straight/distinct_straight_num_1_structure.xyz"
# %run sorting_structures.py -i reactant_state_pcracking/structures.xyz -ms 4 -o reactant_state_pcracking/sorted_struct.txt
import numpy as np
import argparse
import os
import collections


def count_lines(readf):
    return len(readf)


def extract_energy(linecount, mol_size, readf, start):
    c = 0
    energy_array = np.zeros((int((linecount - start) / (mol_size + start)) + 1))
    for i in range(start - 1, linecount, mol_size + start):
        whole_line = readf[i]
        line_array = [
            float(item) if item[0] != "*" else float("Nan")
            for item in whole_line.split()
        ]
        energy_array[c] = line_array[0]
        c += 1
    return energy_array


def merge_files(file1, file2, start, m1, out_file):
    r1 = read_file(file1)
    r2 = read_file(file2)
    l1 = int(count_lines(r1) / (m1 + start))
    fileout = open(out_file, "w")
    fileout.truncate(0)
    for i in range(l1 - 1):
        print("9", end=" \n", file=fileout)
        [
            print(str(r1[j]), end=" ", file=fileout)
            for j in range(i * (m1 + start) + 1, (i + 1) * (m1 + start))
        ]
        [
            print(str(r2[j]), end=" ", file=fileout)
            for j in range(i * (5 + start) + start, (i + 1) * (5 + start))
        ]
    fileout.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Sample lower energy structures")
    parser.add_argument("-i1", "--inp_file_1", action="store", default="fort.102")
    parser.add_argument("-i2", "--inp_file_2", action="store", default="system.xyz")
    parser.add_argument("-ms", "--mol_size", action="store", default=8)
    parser.add_argument("-o", "--out_file", action="store", default="output.txt")
    args, unknown = parser.parse_known_args()
    return args


def read_file(fname):
    openf = open(fname)
    readf = openf.readlines()
    return readf


def write_sorted_configurations_1(out_file, index, readf, mol_size, start):
    fileout = open(out_file, "w")
    fileout.truncate(0)
    nindex = index * (mol_size + start)
    for i in nindex:
        for j in range(i, i + mol_size + start, 1):
            print(str(readf[j]), end=" ", file=fileout)
    fileout.close()


def write_sorted_configurations_2(out_file, index, readf, mol_size, start):
    fileout = open(out_file, "w")
    fileout.truncate(0)
    nindex = index * (mol_size + start)
    for i in nindex:
        print(str(readf[i]), end=" ", file=fileout)
        print("", file=fileout)
        for j in range(i + mol_size + start - 5, i + mol_size + start, 1):
            print(str(readf[j]), end=" ", file=fileout)
    fileout.close()


if __name__ == "__main__":

    args = parse_args()
    print(args)

    start = 2
    readf = read_file(args.inp_file_1)
    linecount = count_lines(readf)
    energy_array = extract_energy(linecount, int(args.mol_size), readf, start)
    index_sorted = np.argsort(energy_array)
    sorted_energy_array = energy_array[index_sorted]
    if args.inp_file_2 == "system.xyz":
        sys_readf = read_file(args.inp_file_2)
        sys_linecount = count_lines(readf)
        sys_size = int(sys_readf[0].split()[0])
        write_sorted_configurations_1(
            args.out_file, index_sorted, readf, int(args.mol_size), start
        )
        write_sorted_configurations_2(
            "sorted_system.xyz", index_sorted, sys_readf, sys_size, start
        )
        merge_files(
            args.out_file,
            "sorted_system.xyz",
            start,
            int(args.mol_size),
            "final_struct.xyz",
        )
    else:
        write_sorted_configurations_1(
            args.out_file, index_sorted, readf, int(args.mol_size), start
        )
    # input_file = read_file(args.out_file)
    # zeolite_name = "FAU-2"

    print("done!")
