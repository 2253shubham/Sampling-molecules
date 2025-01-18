# %run sampling_top_structures_for_all_O_sites.py -i1 zeolite_site_info/LTA-2/LTA-2.cif -i2 LTA-2/TS/sorted_struct.txt -ms 8 -lx 24.555 -ly 24.555 -lz 24.555 -ns 1 -co 0.5
# %run ../../../../sampling_top_structures_for_all_O_sites.py -i1 ../../../../zeolite_site_info/TON-1/TON-1.cif -i2 sorted_struct_old.txt -ms 7 -lx 13.859 -ly 17.420 -lz 5.038 -ns 20 -co 0.5 -ang 0 -dis 1.55 -gh 0 #-zs 0

# needed parameters (all dis. in angs., all angles in degrees)
# BS/TS --> distance = 2.17, angle = 165, consider C-O bond distance (system size: 7 for C2C2, 8 for C2C3, 9 for C2mC3)
# HT/TS --> distance = 2.27, consider C-O bond distance (system size: 9 for all)
# BS/RS --> C-O bond distance considered (carbon number varies with system types as specified) (1.52 for C2C2, 1.56 for C2C3, 1.66 for C2mC3) (7 for C2C2, 8 for C2C3, 9 for C2mC3)
# HT/RS --> C-O bond distance considered (fixed across all different types) (1.52)


import numpy as np
import argparse
import os
import collections
from pymatgen import Structure, Lattice, Molecule
from pymatgen.io.cif import CifWriter
from pymatgen.io.xyz import XYZ
from openbabel import openbabel, pybel
from ase.build.supercells import make_supercell  # required for TON
from pymatgen.util.coord import pbc_diff


def center_of_mass_data_and_hydrogen_coor_and_energies(
    readf, linecount, mol_size, start, elements, counter, gh, rxn
):  # modified to C-O bond distance for TS
    c = 1
    com_coor_data = []
    energies = []
    hydrogen_coor = []
    n1 = []
    n2 = []
    for i in range(0, linecount - 1, mol_size + start):
        coordinates = np.array(
            [
                [float(item) for item in whole_line.split()[1:4]]
                for whole_line in readf[i + start : c * (mol_size + start)]
            ]
        )
        energies.append(np.float(readf[i + start - 1].split()[0]))
        # com_coor = com(coordinates, elements) # without surface atoms
        if counter == 1:
            com_coor = coordinates[
                len(coordinates) - 3 - 1
            ]  # if C-O distance is considered
            # mol_coord = coordinates[:len(coordinates)-3] # if com distance is considered
            # com_coor = com(mol_coord, elements) # if com distance is considered
            com_coor_data.append(com_coor)
            n1.append(coordinates[len(coordinates) - 3 - 3])
            hydrogen_coor.append(coordinates[4])
        elif counter > 1 and gh == 0 and rxn == "bs":
            com_coor = com(
                coordinates[: len(coordinates) - 3], elements
            )  # without surface atoms
            com_coor_data.append(com_coor)
            hydrogen_coor.append(
                coordinates[len(coordinates) - 3 - 3]
            )  # actually carbon atom (change accordingly) (1 for C2C2, 2 for C2C3, 3 for C2mC3)
            n1.append(
                coordinates[len(coordinates) - 3 - 4]
            )  # actually carbon atom (change accordingly) (2 for C2C2, 3 for C2C3, 4 for C2mC3)
        elif counter > 1 and gh == 0 and rxn == "ht":
            com_coor = com(
                coordinates[: len(coordinates) - 5], elements
            )  # without surface atoms (only the adsorbed molecule)
            com_coor_data.append(com_coor)
            hydrogen_coor.append(
                coordinates[len(coordinates) - 3 - 1]
            )  # fixed carbon atom for all systems
            n1.append(coordinates[len(coordinates) - 3 - 2])
        else:
            com_coor = com(coordinates, elements)  # without surface atoms
            com_coor_data.append(com_coor)
            hydrogen_coor.append(coordinates[4])
        # mol_coord = coordinates[:len(coordinates)-3]
        # com_coor = com(mol_coord, elements)
        # com_coor_data.append(com_coor)
        # hydrogen_coor.append(coordinates[4])
        c += 1
    return (
        np.array(com_coor_data).reshape(-1, 3),
        np.array(energies).reshape(-1, 1),
        np.array(hydrogen_coor).reshape(-1, 3),
        np.array(n1).reshape(-1, 3),
    )


def clone(filename):
    mol = pybel.readfile("cif", filename).__next__()
    cm = mol.clone
    return cm


def closest(
    top_O_sites,
    ns,
    com_data,
    wcom_data,
    O_coor,
    wO_coor,
    si_coor,
    lattice,
    cut_off,
    counter_array,
    com_counter_array,
    filename1,
    filename2,
    n1,
    ref_angle,
    ref_dis,
    h_coor,
    gh,
    rxn,
    zeo,
):
    min_dis = 100
    prox = []
    min_dis_index = 0
    f = 100
    for i in range(len(O_coor)):
        angles = []
        # print(O_coor[i])

        diff = np.linalg.norm(
            pbc_diff(com_data / lattice.T, O_coor[i] / lattice.T) * lattice.T, axis=1
        )
        diff_rs = np.linalg.norm(
            pbc_diff(h_coor / lattice.T, O_coor[i] / lattice.T) * lattice.T, axis=1
        )
        # print(diff_rs)
        d1 = pbc_diff(n1 / lattice.T, com_data / lattice.T) * lattice.T
        d2 = pbc_diff(O_coor[i] / lattice.T, com_data / lattice.T) * lattice.T
        for j in range(len(O_coor[i])):
            cosine_angle = np.dot(d1, d2[j]) / (
                np.linalg.norm(d1) * np.linalg.norm(d2[j])
            )
            angles.append(np.degrees(np.arccos(cosine_angle)))
        tmp_a = np.array(angles) - ref_angle
        tmp_dis = diff - ref_dis
        tmp_dis_rs = diff_rs - ref_dis
        # tmp_dis_rs = diff
        # print(tmp_dis_rs)
        # print(np.min(np.abs(tmp_a)), np.min(np.abs(tmp_dis)))
        # print(tmp_a[0:100])
        if ns == 1:
            if rxn == "bs":  # needs angle and bond distance
                if (np.min(np.abs(tmp_a)) < 0.01) and (np.min(np.abs(tmp_dis)) < 0.01):
                    # print(np.min(np.abs(tmp_a)), np.min(np.abs(tmp_dis)))
                    min_dis_index = np.argmin(np.abs(tmp_a))
                    # print(min_dis_index)
                    min_dis = diff[min_dis_index]
                    # print(min_dis)
                    f = i
            else:
                if np.min(np.abs(tmp_dis)) < 0.01:
                    # print(np.min(np.abs(tmp_a)), np.min(np.abs(tmp_dis)))
                    min_dis_index = np.argmin(np.abs(tmp_dis))
                    # print(min_dis_index)
                    min_dis = diff[min_dis_index]
                    # print(min_dis)
                    f = i
        elif ns > 1 and gh == 0 and rxn == "bs":
            min_dis_index_temp = np.argmin(np.abs(diff_rs))
            if tmp_dis_rs[min_dis_index_temp] < 0.005:
                # print(np.min(np.abs(tmp_a)), np.min(np.abs(tmp_dis)))
                # min_dis_index = np.argmin(np.abs(diff))
                # print(min_dis_index)
                min_dis_index = min_dis_index_temp
                min_dis = diff_rs[min_dis_index]
                # print(min_dis)
                f = i
        elif ns > 1 and gh == 0 and rxn == "ht":
            # print("entered_correct_loop")
            tmp = np.min(np.abs(diff_rs))
            if min_dis > tmp:
                min_dis = tmp
                min_dis_index = np.argmin(np.abs(diff_rs))
                f = i
        else:
            #            min_dis_index = np.argmin(np.abs(diff))
            #            min_dis = diff[min_dis_index]
            #            f = i
            tmp = np.min(np.abs(diff))
            if min_dis > tmp:
                min_dis = tmp
                min_dis_index = np.argmin(np.abs(diff))
                f = i
    print(min_dis)
    if min_dis != 100:
        if counter_array[f] < ns:
            if zeo == "TON":
                cm = clone(filename2)
                [
                    cm.OBMol.DeleteAtom(cm.atoms[i].OBAtom)
                    for i in range(len(cm.atoms))[::-1]
                ]
                a1 = openbabel.OBAtom()
                a1.SetVector(float(com_data[0]), float(com_data[1]), float(com_data[2]))
                a1.SetAtomicNum(8)
                cm.OBMol.AddAtom(a1)
                cm.unitcell.FillUnitCell(cm.OBMol)
                cm.write("xyz", "mol_grown.xyz", overwrite=True)
            else:
                cm = clone(filename1)
                [
                    cm.OBMol.DeleteAtom(cm.atoms[i].OBAtom)
                    for i in range(len(cm.atoms))[::-1]
                ]
                a1 = openbabel.OBAtom()
                a1.SetVector(float(com_data[0]), float(com_data[1]), float(com_data[2]))
                a1.SetAtomicNum(8)
                cm.OBMol.AddAtom(a1)
                cm.unitcell.FillUnitCell(cm.OBMol)
                cm.write("xyz", "mol_grown.xyz", overwrite=True)
            ext_coord = np.loadtxt(fname="mol_grown.xyz", skiprows=2, usecols=(1, 2, 3))
            if zeo == "LTA" and rxn == "ht":
                if screen_atoms(ext_coord / lattice.T):
                    return -1, 0, 0, 0, 0
            for i in com_counter_array[f]:
                prox.append(np.linalg.norm(i - ext_coord, axis=1))
            if np.min(prox) > cut_off:
                # directory = "distinct_sites/"
                # if not os.path.exists(directory):
                # 	os.makedirs(directory)
                # out = open("distinct_sites/site_data.txt","a")
                out = open("site_data.txt", "a")  # only for ht
                print("got a structure at :", file=out)
                print(top_O_sites[f], file=out)
                print("com of the found structure", file=out)
                print(com_data, file=out)
                print("location of oxygen at the site", file=out)
                print("C location", file=out)
                print(h_coor, file=out)
                print("O location", file=out)
                print(O_coor[f][min_dis_index], file=out)
                print("distance", file=out)
                print(min_dis, file=out)
                counter_array[f] += 1
                print("structure number", file=out)
                print(int(counter_array[f]), file=out)
                out.close()
                print("got a structure at :")
                print(top_O_sites[f])
                print("com of the found structure")
                print(com_data)
                print("C location")
                print(h_coor)
                print("min_dis_index")
                print(min_dis_index)
                print("location of oxygen at the site")
                print(O_coor[f][min_dis_index])
                print("distance")
                print(min_dis)
                print("structure number")
                print(int(counter_array[f]))
                com_counter_array[f, int(counter_array[f]) - 1] = com_data
                diff_sio = np.linalg.norm(
                    si_coor - np.array(O_coor[f][min_dis_index]), axis=1
                )
                min_dsio1 = np.min(diff_sio)
                sic1 = np.argmin(diff_sio)
                diff_sio[diff_sio == min_dsio1] = 10000000
                min_dsio2 = np.min(diff_sio)
                sic2 = np.argmin(diff_sio)
                si1 = si_coor[sic1]
                si2 = si_coor[sic2]
                omid = np.array(O_coor[f][min_dis_index])
                v1 = omid - si1
                if abs(min_dsio1 - min_dsio2) < 0.5:
                    v2 = omid - si2
                else:
                    print("***doing modification for h grow for this site***")
                    min1 = np.min(abs(omid - lattice.T))
                    dim1 = np.argmin(abs(omid - lattice.T))
                    min2 = np.min(omid)
                    dim2 = np.argmin(omid)
                    x = si_coor
                    if min1 > min2:
                        flag = 2
                        y = extend_single_cell_si_atoms(
                            dim2, np.array(x), lattice, flag
                        )
                    else:
                        flag = 1
                        y = extend_single_cell_si_atoms(
                            dim1, np.array(x), lattice, flag
                        )
                    diff_sio = np.linalg.norm(
                        y - np.array(O_coor[f][min_dis_index]), axis=1
                    )
                    min_dsio2 = np.min(diff_sio)
                    sic2 = np.argmin(diff_sio)
                    si2 = y[sic2]
                    v2 = omid - si2

                    """                
                    v2=v1
                    close_to_lat1 = np.linalg.norm(omid-lattice.T, axis=0)
                    close_to_lat2 = omid
                    if (np.min(close_to_lat1)>np.min(close_to_lat2)):
                        close_to_lat = close_to_lat2
                    else:
                        close_to_lat = close_to_lat1
                    index = np.argmin(close_to_lat)
                    v2[index] = -v2[index]
                    """
                v3 = v1 + v2
                h_loc1 = omid - (v3 / np.linalg.norm(v3))
                d1 = np.linalg.norm(
                    pbc_diff(h_loc1 / lattice.T, com_data / lattice.T) * lattice.T
                )
                h_loc2 = omid + (v3 / np.linalg.norm(v3))
                d2 = np.linalg.norm(
                    pbc_diff(h_loc2 / lattice.T, com_data / lattice.T) * lattice.T
                )
                if d1 > d2:
                    h_loc = h_loc2
                else:
                    h_loc = h_loc1
                return f, min_dis_index, h_loc, si1, O_coor[f][min_dis_index]
            else:
                return -1, 0, 0, 0, 0
        else:
            return -1, 0, 0, 0, 0
    else:
        return -1, 0, 0, 0, 0


def com(coordinates, elements):
    mydict = {"C": 12, "O": 16, "H": 1}
    c = collections.Counter(elements)
    totmass = np.sum([c[i] * mydict[i] for i in list(mydict)])
    mwc = np.sum(
        [coordinates[i] * mydict[elements[i]] for i in range(len(coordinates))], axis=0
    )
    return mwc / totmass


def create_cif_file(lattice, coor):
    my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]]])
    structure = Structure(
        my_lattice,
        ["C", "C", "C", "C"],
        [coor[0], coor[1], coor[2], coor[3]],
        coords_are_cartesian=True,
    )
    j = CifWriter(structure)
    j.write_file("temp_cif.cif")


def create_cif_file_of_samp_molecule(
    filename, lattice, final_coor, elements, str_num, name, cut_off, gh, oc
):
    my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]]])
    # coord = retransform(final_coor, lattice)
    molecule = Molecule.from_file(filename)
    coord_f = molecule.cart_coords
    # coord = retransform(final_coor, lattice)
    # coord_nf = np.concatenate((coord_f, coord))
    coord_nf = np.concatenate((coord_f, final_coor))
    mol_size = len(final_coor)
    elements_f = np.array(molecule.species)
    str_elements = [str(elements_f[i]) for i in range(len(elements_f))]
    all_elements = str_elements + elements
    """
    if (gh==0):
        #ocr = retransform(oc.reshape(-1,3), lattice)
        structure = Structure(elements, final_coor, coords_are_cartesian=True)
        j = CifWriter(structure)
        j.write_file("temp_final_cif.cif")
        mol = pybel.readfile("cif", "temp_final_cif.cif").__next__()
        mol.write("pdb", "temp_final_pdb.pdb" , overwrite=True)
        mol = pybel.readfile("pdb", "temp_final_pdb.pdb").__next__()
        mol.addh()
        mol.write("pdb", "temp_final_pdb.pdb" , overwrite=True)
        molecule2 = Molecule.from_file("temp_final_pdb.pdb")
        coord = molecule2.cart_coords
        diff = np.linalg.norm(ocr-coord, axis=1)
        min_index = np.argmin(np.abs(diff))
        coord = np.delete(coord, min_index, 0)
        coord_nf = np.concatenate((coord_f, coord))
        elements = np.array(molecule2.species)
        elements = np.delete(elements, min_index, 0)
        st_elements = [str(elements[i]) for i in range(len(elements))]
        all_elements = str_elements + st_elements
    """
    my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]]])
    coord_fn = retransform(coord_nf, lattice)
    structure = Structure(my_lattice, all_elements, coord_fn, coords_are_cartesian=True)
    j = CifWriter(structure)
    directory = (
        "distinct_sites/" + str(cut_off) + "/" + str(name) + "/"
    )  # change this +"C-O-dis"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # j.write_file(directory+"structure_num_"+str(int(str_num))+"_of_butane_TS.cif")
    j.write_file(directory + "structure_num_" + str(int(str_num)) + "_of_butane_RS.cif")


def create_final_pdb_with_zeolite(
    tr_coor, sial, bu_elements, lattice, filename, name, cut_off, str_num, energy
):
    molecule = Molecule.from_file(filename)
    coord = molecule.cart_coords
    tr_coor = retransform(tr_coor, lattice)
    l1 = len(tr_coor)
    l2 = len(coord)
    coord_nf = np.concatenate((coord, tr_coor))
    elements = np.array(molecule.species)
    str_elements = [str(elements[i]) for i in range(len(elements))]
    diff = np.linalg.norm(retransform(coord, lattice) - sial, axis=1)
    min_d = np.min(diff)
    ind = np.argmin(diff)
    str_elements[ind] = "Al"
    all_elements = str_elements + bu_elements
    all_elements.append("H")
    coord = retransform(coord_nf, lattice)
    my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]]])
    structure = Structure(my_lattice, all_elements, coord_nf, coords_are_cartesian=True)
    j = CifWriter(structure)
    j.write_file("temp_final_cif.cif")
    mol = pybel.readfile("cif", "temp_final_cif.cif").__next__()
    directory = "distinct_sites/" + str(cut_off) + "/" + str(name) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    out = open(directory + "top_20_sampling_energies.txt", "a")
    print(np.float(energy), file=out)
    out.close()
    print("writing file in")
    print(directory)
    print("*****************************")
    mol.write(
        "pdb",
        directory + "structure_num_" + str(int(str_num)) + "_of_butane.pdb",
        overwrite=True,
    )


def create_xyz_with_added_hydrogens(filename):
    mol = pybel.readfile("cif", filename).__next__()
    b2 = mol.OBMol.GetBond(2, 3)
    b2.SetBondOrder(1)
    mol.write("pdb", "temp_pdb.pdb", overwrite=True)
    mol = pybel.readfile("pdb", "temp_pdb.pdb").__next__()
    b2 = mol.OBMol.GetBond(2, 3)
    b2.SetBondOrder(1)
    mol.atoms[1].OBAtom.SetImplicitHCount(2)
    mol.atoms[2].OBAtom.SetImplicitHCount(2)
    mol.addh()
    mol.write("xyz", "st_with_H.xyz", overwrite=True)


def create_zeolite_cif_and_xyz_file(filename):  # , scaling, lattice):
    mol = pybel.readfile("cif", filename).__next__()
    # if (scaling!=0):
    # com = mol.clone
    # com.remove_atom
    # com.unitcell
    mol.unitcell.FillUnitCell(mol.OBMol)
    mol.write("cif", "osite_info.cif", overwrite=True)
    mol.write("xyz", "osite_info.xyz", overwrite=True)


"""
    else:
        mol.write("cif", "osite_info.cif", overwrite=True)
        mol.write("xyz", "osite_info.xyz", overwrite=True)
        mol2 = Molecule.from_file("osite_info.xyz")
        cdnts = mol2.cart_coords
        elements = [str(mol2.species[i]) for i in range(len(mol2.species))]
        my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]*3]])
        s = Structure(my_lattice, elements, cdnts, coords_are_cartesian=True)
        #s.make_supercell(scaling_matrix=[1,1,3], to_unit_cell=False)
        lattice = make_lattice(float(s.lattice.a), float(s.lattice.b), float(s.lattice.c))
        j = CifWriter(s)
        j.write_file("temp_system_cif.cif")
        mol3 = pybel.readfile("cif", "temp_system_cif.cif").__next__()
        mol3.unitcell.FillUnitCell(mol.OBMol)
        mol3.write("cif", "osite_info.cif", overwrite=True)
        mol3.write("xyz", "osite_info.xyz", overwrite=True)
    return lattice
"""


def energy_parser(
    energy, top_O_sites, com_data, n1, O_coor, lattice, ref_angle, ref_dis, h_coor, rxn
):  # for transition states
    min_dis = 100
    prox = []
    min_dis_index = 0
    f = 100
    for i in range(len(O_coor)):
        angles = []
        diff = np.linalg.norm(
            pbc_diff(com_data / lattice.T, O_coor[i] / lattice.T) * lattice.T, axis=1
        )
        diff_rs = np.linalg.norm(
            pbc_diff(h_coor / lattice.T, O_coor[i] / lattice.T) * lattice.T, axis=1
        )
        d1 = pbc_diff(n1 / lattice.T, com_data / lattice.T) * lattice.T
        d2 = pbc_diff(O_coor[i] / lattice.T, com_data / lattice.T) * lattice.T
        for j in range(len(O_coor[i])):
            cosine_angle = np.dot(d1, d2[j]) / (
                np.linalg.norm(d1) * np.linalg.norm(d2[j])
            )
            angles.append(np.degrees(np.arccos(cosine_angle)))
        tmp_a = np.array(angles) - ref_angle
        # print(np.min(np.abs(tmp_a)))
        tmp_dis = diff - ref_dis
        # print(np.min(np.abs(tmp_dis)))
        tmp_dis_rs = diff_rs - ref_dis

        if rxn == "bs":  # needs angle and bond distance
            if (np.min(np.abs(tmp_a)) < 0.1) and (np.min(np.abs(tmp_dis)) < 0.01):
                min_dis_index = np.argmin(np.abs(tmp_a))
                min_dis = diff[min_dis_index]
                f = i
        else:
            if np.min(np.abs(tmp_dis)) < 0.01:
                min_dis_index = np.argmin(np.abs(tmp_dis))
                min_dis = diff[min_dis_index]
                f = i
    oute = open(top_O_sites[f] + "-energy.txt", "a")
    print(float(energy * 8.314 / 1000), file=oute)  # in kJ/mol
    oute.close()


def exec_code(
    freqIntersection,
    freqSinusoidal,
    freqStraight,
    readf,
    mol_size,
    num_of_struct,
    start,
    elements,
    energy_array,
    reference_i_com,
    reference_z_com,
    reference_s_com,
):
    c = 1
    fx = 0
    fy = 0
    fz = 0
    di = 1
    ds = 1
    dz = 1
    out1 = open("straight_ener.txt", "w")
    out2 = open("sinusoidal_ener.txt", "w")
    out3 = open("intersection_ener.txt", "w")
    out4 = open("distinct_straight_com.txt", "w")
    out5 = open("distinct_sinusoidal_com.txt", "w")
    out6 = open("distinct_intersection_com.txt", "w")
    out1.truncate(0)
    out2.truncate(0)
    out3.truncate(0)
    out4.truncate(0)
    out5.truncate(0)
    out6.truncate(0)
    for i in range(0, num_of_struct * (mol_size + start) - 1, mol_size + start):
        a = freqIntersection
        b = freqSinusoidal
        d = freqStraight
        coordinates = np.array(
            [
                [float(item) for item in whole_line.split()[1:4]]
                for whole_line in readf[i + start : c * (mol_size + start)]
            ]
        )
        mol_coord = coordinates[: len(coordinates) - 3]
        com_coor = com(coordinates, elements)  # without surface atoms
        # com_coor = com(mol_coord, elements)
        freqIntersection, freqSinusoidal, freqStraight = siting(
            freqIntersection, freqSinusoidal, freqStraight, com_coor
        )
        print(freqIntersection, freqSinusoidal, freqStraight)
        if freqIntersection == 1 and fx == 0:
            uc_coord = [
                unit_cell_coordinates(coordinates[i]) for i in range(len(coordinates))
            ]
            write_top_configurations(
                readf,
                mol_size,
                "top_freq_intersection.xyz",
                int(i / (mol_size + start)),
                coord=uc_coord,
            )
            fx = 1
        if freqSinusoidal == 1 and fy == 0:
            uc_coord = [
                unit_cell_coordinates(coordinates[i]) for i in range(len(coordinates))
            ]
            write_top_configurations(
                readf,
                mol_size,
                "top_freq_sinusoidal.xyz",
                int(i / (mol_size + start)),
                coord=uc_coord,
            )
            fy = 1
        if freqStraight == 1 and fz == 0:
            uc_coord = [
                unit_cell_coordinates(coordinates[i]) for i in range(len(coordinates))
            ]
            write_top_configurations(
                readf,
                mol_size,
                "top_freq_straight.xyz",
                int(i / (mol_size + start)),
                coord=uc_coord,
            )
            fz = 1
        if freqIntersection - a > 0:
            print(energy_array[int(i / (mol_size + start))], file=out3)
            max_dis = max_dis_between_two_structures(
                reference_i_com, np.array(unit_cell_coordinates(com_coor))
            )
            print(max_dis)
            if max_dis < 5 and di <= 20:
                out6 = open("distinct_intersection_com.txt", "a")
                print(*unit_cell_coordinates(com_coor), file=out6)
                out6.close()
                fname = "distinct_intersection_num_" + str(di) + "_structure.xyz"
                uc_coord = [
                    unit_cell_coordinates(coordinates[i])
                    for i in range(len(coordinates))
                ]
                write_top_configurations(
                    readf, mol_size, fname, int(i / (mol_size + start)), coord=uc_coord
                )
                di += 1
        if freqSinusoidal - b > 0:
            print(energy_array[int(i / (mol_size + start))], file=out2)
            max_dis = max_dis_between_two_structures(
                reference_z_com, np.array(unit_cell_coordinates(com_coor))
            )
            print(max_dis)
            if max_dis < 1.5 and dz <= 20:
                out5 = open("distinct_sinusoidal_com.txt", "a")
                print(*unit_cell_coordinates(com_coor), file=out5)
                out5.close()
                fname = "distinct_sinusoidal_num_" + str(dz) + "_structure.xyz"
                uc_coord = [
                    unit_cell_coordinates(coordinates[i])
                    for i in range(len(coordinates))
                ]
                write_top_configurations(
                    readf, mol_size, fname, int(i / (mol_size + start)), coord=uc_coord
                )
                dz += 1
        if freqStraight - d > 0:
            print(energy_array[int(i / (mol_size + start))], file=out1)
            max_dis = max_dis_between_two_structures(
                reference_s_com, np.array(unit_cell_coordinates(com_coor))
            )
            print(max_dis)
            if max_dis < 10 and ds <= 20:
                out4 = open("distinct_straight_com.txt", "a")
                print(*unit_cell_coordinates(com_coor), file=out4)
                out4.close()
                fname = "distinct_straight_num_" + str(ds) + "_structure.xyz"
                uc_coord = [
                    unit_cell_coordinates(coordinates[i])
                    for i in range(len(coordinates))
                ]
                write_top_configurations(
                    readf, mol_size, fname, int(i / (mol_size + start)), coord=uc_coord
                )
                ds += 1
        c += 1
    out1.close()
    out2.close()
    out3.close()
    out4.close()
    out5.close()
    out6.close()
    return freqIntersection, freqSinusoidal, freqStraight


def extend_single_cell_si_atoms(dim, si_coor, lattice, flag):
    if flag == 1:
        si_coor[:, dim] = si_coor[:, dim] + lattice[dim]
    else:
        si_coor[:, dim] = si_coor[:, dim] - lattice[dim]
    return si_coor


def extract_atom_indices(array):
    elements = [array[i].split()[0] for i in range(len(array))]
    return elements


def extract_o_lines(fname):
    with open(fname) as f:
        with open("o_out.txt", "w") as f1:
            for line in f:
                if "O " in line:
                    f1.write(line)
    f1 = np.loadtxt(fname="o_out.txt", usecols=(2, 3, 4))
    f2 = np.loadtxt(fname="o_out.txt", dtype=str, usecols=0)
    return f1, f2


def extract_si_lines(fname):
    with open(fname) as f:
        with open("si_out.txt", "w") as f1:
            for line in f:
                if "Si" in line:
                    f1.write(line)
    f1 = np.loadtxt(fname="si_out.txt", usecols=(2, 3, 4))
    f2 = np.loadtxt(fname="si_out.txt", dtype=str, usecols=0)
    return f1, f2


def extract_pos(array, element):
    pos = []
    for i in range(len(array)):
        if array[i] == element:
            pos.append(i)
    return np.array(pos)


def grow_hydrogen_to_reactant_state_bs(
    filename, lattice, final_coor, elements, str_num, name, cut_off, gh, oc, sial
):
    my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]]])
    my_lattice_2 = Lattice(
        [[lattice[0] * 2, 0, 0], [0, lattice[1] * 2, 0], [0, 0, lattice[2] * 2]]
    )
    # coord = retransform(final_coor, lattice)
    molecule = Molecule.from_file(filename)
    coord_f = molecule.cart_coords
    # coord = retransform(final_coor, lattice)
    # coord_nf = np.concatenate((coord_f, coord))
    diff = np.linalg.norm(retransform(coord_f, lattice) - sial, axis=1)
    min_d = np.min(diff)
    ind = np.argmin(diff)
    mol_size = len(final_coor)
    print(mol_size)
    elements_f = np.array(molecule.species)
    str1_elements = [str(elements_f[i]) for i in range(len(elements_f))]
    str1_elements[ind] = "Al"
    # all_elements = str_elements + elements
    # ocr = retransform(oc.reshape(-1,3), lattice)
    structure = Structure(my_lattice_2, elements, final_coor, coords_are_cartesian=True)
    j = CifWriter(structure)
    j.write_file("temp_final_cif_adsorbate.cif")
    mol = pybel.readfile("cif", "temp_final_cif_adsorbate.cif").__next__()
    #    for i in range(mol_size-1):
    #        b2 = mol.OBMol.GetBond(i+1,i+2)
    #        b2.SetBondOrder(1)
    mol.write("pdb", "temp_pdb_adsorbate.pdb", overwrite=True)
    mol = pybel.readfile("pdb", "temp_pdb_adsorbate.pdb").__next__()
    for i in range(mol_size - 2):
        b2 = mol.OBMol.GetBond(i + 1, i + 2)
        b2.SetBondOrder(1)
    mol.atoms[0].OBAtom.SetImplicitHCount(3)
    mol.atoms[1].OBAtom.SetImplicitHCount(2)
    mol.atoms[2].OBAtom.SetImplicitHCount(2)
    if mol_size == 4:
        mol.atoms[3].OBAtom.SetImplicitHCount(3)
    elif mol_size == 5:
        mol.atoms[3].OBAtom.SetImplicitHCount(2)
    else:
        mol.atoms[3].OBAtom.SetImplicitHCount(1)
    mol.addh()
    mol.write("pdb", "temp_final_pdb_adsorbate.pdb", overwrite=True)
    # mol.write("xyz", "st_with_H.xyz", overwrite=True)
    mol = pybel.readfile("pdb", "temp_final_pdb_adsorbate.pdb").__next__()
    molecule2 = Molecule.from_file("temp_final_pdb_adsorbate.pdb")
    coord = molecule2.cart_coords
    coord = retransform(coord, lattice)
    print("all_coord")
    print(coord)
    diff = np.linalg.norm(
        pbc_diff(oc / lattice.T, coord / lattice.T) * lattice.T, axis=1
    )
    print("diff")
    print(diff)
    min_index = np.argmin(np.abs(diff))
    print("oc")
    print(oc)
    print("coord_min_H")
    print(coord[min_index])
    st_elements = np.array(molecule2.species)
    if np.min(diff) <= float(args.ref_dis):
        coord = np.delete(coord, min_index, 0)
        st_elements = np.delete(st_elements, min_index, 0)
    coord_nf = np.concatenate((coord_f, coord))
    str2_elements = [str(st_elements[i]) for i in range(len(st_elements))]
    all_elements = str1_elements + str2_elements
    my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]]])
    coord_fn = retransform(coord_nf, lattice)
    structure = Structure(my_lattice, all_elements, coord_fn, coords_are_cartesian=True)
    j = CifWriter(structure)
    directory = (
        "distinct_sites/" + str(cut_off) + "/" + str(name) + "/"
    )  # change this +"C-O-dis"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # j.write_file(directory+"structure_num_"+str(int(str_num))+"_of_butane_TS.cif")
    j.write_file(directory + "structure_num_" + str(int(str_num)) + "_of_bs_RS.cif")


def grow_hydrogen_to_reactant_state_ht(
    filename, lattice, final_coor, elements, str_num, name, cut_off, gh, oc, sial
):
    my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]]])
    my_lattice_2 = Lattice(
        [[lattice[0] * 2, 0, 0], [0, lattice[1] * 2, 0], [0, 0, lattice[2] * 2]]
    )
    # coord = retransform(final_coor, lattice)
    molecule = Molecule.from_file(filename)
    coord_f = molecule.cart_coords
    # coord = retransform(final_coor, lattice)
    # coord_nf = np.concatenate((coord_f, coord))
    diff = np.linalg.norm(retransform(coord_f, lattice) - sial, axis=1)
    min_d = np.min(diff)
    ind = np.argmin(diff)
    mol_size = len(final_coor)
    print(mol_size)
    elements_f = np.array(molecule.species)
    str1_elements = [str(elements_f[i]) for i in range(len(elements_f))]
    str1_elements[ind] = "Al"
    # all_elements = str_elements + elements
    # ocr = retransform(oc.reshape(-1,3), lattice)
    structure = Structure(my_lattice_2, elements, final_coor, coords_are_cartesian=True)
    j = CifWriter(structure)
    j.write_file("temp_final_cif.cif")
    mol = pybel.readfile("cif", "temp_final_cif.cif").__next__()
    #    for i in range(mol_size-1):
    #        b2 = mol.OBMol.GetBond(i+1,i+2)
    #        b2.SetBondOrder(1)
    mol.write("pdb", "temp_pdb.pdb", overwrite=True)
    mol = pybel.readfile("pdb", "temp_pdb.pdb").__next__()
    #    for i in range(mol_size-2):
    #        b2 = mol.OBMol.GetBond(i+1,i+2)
    #        b2.SetBondOrder(1)
    # mol.atoms[4].OBAtom.SetImplicitHCount(3)
    # mol.atoms[5].OBAtom.SetImplicitHCount(3)

    for i in range(mol_size):
        mol.atoms[i].OBAtom.SetHyb(3)
    mol.addh()
    mol.write("pdb", "temp_final_pdb.pdb", overwrite=True)
    # mol.write("xyz", "st_with_H.xyz", overwrite=True)
    mol = pybel.readfile("pdb", "temp_final_pdb.pdb").__next__()
    molecule2 = Molecule.from_file("temp_final_pdb.pdb")
    coord = molecule2.cart_coords
    coord = retransform(coord, lattice)
    print("all_coord")
    print(coord)
    diff = np.linalg.norm(oc - coord, axis=1)
    print("diff")
    print(diff)
    min_index = np.argmin(np.abs(diff))
    print("oc")
    print(oc)
    print("coord_min_H")
    print(coord[min_index])
    coord = np.delete(coord, min_index, 0)
    coord_nf = np.concatenate((coord_f, coord))
    st_elements = np.array(molecule2.species)
    st_elements = np.delete(st_elements, min_index, 0)
    str2_elements = [str(st_elements[i]) for i in range(len(st_elements))]
    all_elements = str1_elements + str2_elements
    my_lattice = Lattice([[lattice[0], 0, 0], [0, lattice[1], 0], [0, 0, lattice[2]]])
    coord_fn = retransform(coord_nf, lattice)
    structure = Structure(my_lattice, all_elements, coord_fn, coords_are_cartesian=True)
    j = CifWriter(structure)
    directory = str(cut_off) + "/" + str(name) + "/"  # change this +"C-O-dis"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # j.write_file(directory+"structure_num_"+str(int(str_num))+"_of_butane_TS.cif")
    j.write_file(directory + "structure_num_" + str(int(str_num)) + "_of_ht_RS.cif")


def make_lattice(x0, y0, z0):
    return np.array([x0, y0, z0]).reshape(-1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample lower energy structures")
    parser.add_argument("-i1", "--inp1_file", action="store", default="inp1.cif")
    parser.add_argument(
        "-i2", "--inp2_file", action="store", default="sorted_struct.txt"
    )
    parser.add_argument("-i3", "--inp3_file", action="store", default="inp2.cif")
    parser.add_argument("-ms", "--mol_size", action="store", default=4)
    parser.add_argument("-lx", "--lattice_x", action="store", default=24.555)
    parser.add_argument("-ly", "--lattice_y", action="store", default=24.555)
    parser.add_argument("-lz", "--lattice_z", action="store", default=24.555)
    parser.add_argument("-ns", "--num_of_struct", action="store", default=20)
    parser.add_argument("-co", "--cut_off", action="store", default="0.5")
    parser.add_argument("-ang", "--ref_angle", action="store", default="165")
    parser.add_argument(
        "-dis", "--ref_dis", action="store", default="2"
    )  # change for bs (1.52 for C2C2, 1.56 for C2C3, 1.66 for C2mC3), for ht (fixed 1.52)
    parser.add_argument(
        "-gh", "--grow_h", action="store", default="1"
    )  # 0 to grow hydrogens, 1 for not to grow hydrogens
    parser.add_argument(
        "-rxn", "--rxn_type", action="store", default="bs"
    )  # bs = beta scission, ht = hydride transfer
    parser.add_argument("-zeo", "--zeo_name", action="store", default="LTA")
    #   parser.add_argument('-zs', '--zeo_sc', action='store', default='1')
    args, unknown = parser.parse_known_args()
    return args


def retransform(array, lattice):
    for j in range(3):
        while any(array[:, j] < 0) or any(array[:, j] > lattice[j]):
            array[:, j][array[:, j] < 0] = array[:, j][array[:, j] < 0] + lattice[j]
            array[:, j][array[:, j] > lattice[j]] = (
                array[:, j][array[:, j] > lattice[j]] - lattice[j]
            )
    return array


def screen_atoms(coords):
    x_range = [0.3, 0.7]
    y_range = [0.3, 0.7]
    z_range = [0.3, 0.7]

    # Check if any coordinate set lies within the range
    within_range = np.any(
        (coords[:, 0] >= x_range[0])
        & (coords[:, 0] <= x_range[1])
        & (coords[:, 1] >= y_range[0])
        & (coords[:, 1] <= y_range[1])
        & (coords[:, 2] >= z_range[0])
        & (coords[:, 2] <= z_range[1])
    )

    return within_range


def siting(freqIntersection, freqSinusoidal, freqStraight, coord):
    x0 = 20.022
    y0 = 19.899
    z0 = 13.383
    r0sq = 2.65**2
    dy1 = coord[1] - 5.06
    dy1 = dy1 - round(dy1 / y0) * y0
    dy2 = coord[1] - 14.84
    dy2 = dy2 - round(dy2 / y0) * y0

    dx1 = coord[0] - 0.01
    dx1 = dx1 - round(dx1 / x0) * x0
    dz1 = coord[2] - 6.65
    dz1 = dz1 - round(dz1 / z0) * z0
    sxz = dx1**2 + dz1**2
    if sxz <= r0sq:  # in straight channel (0.01,y,6.65)
        # in intersection (0.01,5.06 or 14.84,6.65)
        if sxz + dy1**2 <= r0sq or sxz + dy2**2 <= r0sq:
            freqIntersection += 1
        else:
            freqStraight += 1
    else:
        dx1 = coord[0] - 10.02
        dx1 = dx1 - round(dx1 / x0) * x0
        dz1 = coord[2] - 0.04
        dz1 = dz1 - round(dz1 / z0) * z0
        sxz = dx1**2 + dz1**2
        if sxz <= r0sq:  # in straight channel (10.02,y,0.04)
            # in intersection (10.02,5.06 or 14.84,0.04)
            if sxz + dy1**2 <= r0sq or sxz + dy2**2 <= r0sq:
                freqIntersection += 1
            else:
                freqStraight += 1
        else:
            freqSinusoidal += 1
    return freqIntersection, freqSinusoidal, freqStraight


def transformations(x, y, z, array, lattice):
    array[0] = np.array([x * lattice[0], y * lattice[1], z * lattice[2]])
    array[1] = np.array(
        [(-x + 0.5) * lattice[0], -y * lattice[1], (z + 0.5) * lattice[2]]
    )
    array[2] = np.array([-x * lattice[0], (y + 0.5) * lattice[1], -z * lattice[2]])
    array[3] = np.array(
        [(x + 0.5) * lattice[0], (-y + 0.5) * lattice[1], (-z + 0.5) * lattice[2]]
    )
    array[4] = np.array([-x * lattice[0], -y * lattice[1], -z * lattice[2]])
    array[5] = np.array(
        [(x + 0, 5) * lattice[0], y * lattice[1], (-z + 0.5) * lattice[2]]
    )
    array[6] = np.array([x * lattice[0], (-y + 0.5) * lattice[1], z * lattice[2]])
    array[7] = np.array(
        [(-x + 0.5) * lattice[0], (y + 0.5) * lattice[1], (z + 0.5) * lattice[2]]
    )
    array[:, 0][array[:, 0] < 0] = array[:, 0][array[:, 0] < 0] + lattice[0]
    array[:, 0][array[:, 0] > lattice[0]] = (
        array[:, 0][array[:, 0] > lattice[0]] - lattice[0]
    )
    array[:, 1][array[:, 1] < 0] = array[:, 1][array[:, 1] < 0] + lattice[1]
    array[:, 1][array[:, 1] > lattice[1]] = (
        array[:, 1][array[:, 1] > lattice[1]] - lattice[1]
    )
    array[:, 2][array[:, 2] < 0] = array[:, 2][array[:, 2] < 0] + lattice[2]
    array[:, 2][array[:, 2] > lattice[2]] = (
        array[:, 2][array[:, 2] > lattice[2]] - lattice[2]
    )
    return array


def unit_cell_coordinates(coord, lattice):
    mod_coord = [
        [coord1[i] * float(lattice[i]) for i in range(len(coord1))] for coord1 in coord
    ]
    coord_wrt_lat = [
        [
            (mod_coord1[i] < 0) and float(lattice[i]) + mod_coord1[i] or mod_coord1[i]
            for i in range(len(mod_coord1))
        ]
        for mod_coord1 in mod_coord
    ]
    return coord_wrt_lat


def within_half(coord, lattice):
    mod_coord = [
        [
            (mod_coord1[i] > float(lattice[i]) / 2)
            and mod_coord1[i] - float(lattice[i])
            or mod_coord1[i]
            for i in range(len(mod_coord1))
        ]
        for mod_coord1 in coord
    ]
    return np.array(mod_coord)


if __name__ == "__main__":

    args = parse_args()
    print(args)

    start = 2
    openf = open(args.inp2_file)
    readf = openf.readlines()
    linecount = len(readf)
    lattice = make_lattice(
        float(args.lattice_x), float(args.lattice_y), float(args.lattice_z)
    )
    create_zeolite_cif_and_xyz_file(args.inp1_file)  # , float(args.zeo_sc), lattice1)
    elements = extract_atom_indices(readf[start : start + int(args.mol_size)])
    com_data, energies, hydrogen_coor_data, n1 = (
        center_of_mass_data_and_hydrogen_coor_and_energies(
            readf,
            linecount,
            int(args.mol_size),
            start,
            elements,
            int(args.num_of_struct),
            int(args.grow_h),
            args.rxn_type,
        )
    )
    com_data = retransform(com_data, lattice)
    wcom_data = within_half(com_data, lattice)
    hydrogen_coor_data = retransform(hydrogen_coor_data, lattice)

    lcoor, atom_names = extract_o_lines("osite_info.cif")
    coor = unit_cell_coordinates(lcoor, lattice)
    wcoor = within_half(coor, lattice)
    # coor = retransform(coor, lattice)
    print("O atom names")
    print(atom_names)
    lsi_coor, si_atom_names = extract_si_lines("osite_info.cif")
    si_coor = unit_cell_coordinates(lsi_coor, lattice)
    # si_coor = retransform(si_coor, lattice)
    un_atom_names = np.unique(atom_names)
    print("O atom names unique")
    print(un_atom_names)
    print("*****************************")

    counter_array = np.zeros((len(un_atom_names), 1))
    com_counter_array = np.zeros((len(un_atom_names), int(args.num_of_struct), 3))
    all_com_list = []
    pos = [extract_pos(atom_names, un_atom_names[i]) for i in range(len(un_atom_names))]
    O_coor = np.array(
        [[coor[a[j]] for j in range(len(a))] for a in pos[0 : len(un_atom_names)]]
    )
    wO_coor = np.array(
        [[wcoor[a[j]] for j in range(len(a))] for a in pos[0 : len(un_atom_names)]]
    )

    if int(args.num_of_struct) == 1:
        for i in range(len(com_data)):
            energy_parser(
                energies[i],
                un_atom_names,
                com_data[i],
                n1[i],
                O_coor,
                lattice,
                float(args.ref_angle),
                float(args.ref_dis),
                hydrogen_coor_data[i],
                args.rxn_type,
            )

    for i in range(len(com_data)):
        if int(args.num_of_struct) == 1:
            O_name_index, closest_O_index, h_loc, sial, oc = closest(
                un_atom_names,
                int(args.num_of_struct),
                com_data[i],
                wcom_data[i],
                O_coor,
                wO_coor,
                si_coor,
                lattice,
                float(args.cut_off),
                counter_array,
                com_counter_array,
                args.inp1_file,
                ars.inp3_file,
                n1[i],
                float(args.ref_angle),
                float(args.ref_dis),
                hydrogen_coor_data[i],
                int(args.grow_h),
                args.rxn_type,
                args.zeo_name,
            )
            if O_name_index != -1:
                final_coor = np.array(
                    [
                        [float(item) for item in whole_line.split()[1:4]]
                        for whole_line in readf[
                            (i) * (int(args.mol_size) + start)
                            + start : (i + 1) * (int(args.mol_size) + start)
                        ]
                    ]
                )
                create_cif_file_of_samp_molecule(
                    "osite_info.xyz",
                    lattice,
                    final_coor[: len(final_coor) - 3],
                    elements[: len(elements) - 3],
                    counter_array[O_name_index],
                    un_atom_names[O_name_index],
                    float(args.cut_off),
                    int(args.grow_h),
                    oc,
                )
        elif (
            int(args.num_of_struct) > 1
            and int(args.grow_h) == 0
            and (args.rxn_type == "bs")
        ):
            O_name_index, closest_O_index, h_loc, sial, oc = closest(
                un_atom_names,
                int(args.num_of_struct),
                com_data[i],
                wcom_data[i],
                O_coor,
                wO_coor,
                si_coor,
                lattice,
                float(args.cut_off),
                counter_array,
                com_counter_array,
                args.inp1_file,
                args.inp3_file,
                n1[i],
                float(args.ref_angle),
                float(args.ref_dis),
                hydrogen_coor_data[i],
                int(args.grow_h),
                args.rxn_type,
                args.zeo_name,
            )
            if O_name_index != -1:
                final_coor = np.array(
                    [
                        [float(item) for item in whole_line.split()[1:4]]
                        for whole_line in readf[
                            (i) * (int(args.mol_size) + start)
                            + start : (i + 1) * (int(args.mol_size) + start)
                        ]
                    ]
                )
                grow_hydrogen_to_reactant_state_bs(
                    "osite_info.xyz",
                    lattice,
                    final_coor[: len(final_coor) - 3],
                    elements[: len(elements) - 3],
                    counter_array[O_name_index],
                    un_atom_names[O_name_index],
                    float(args.cut_off),
                    int(args.grow_h),
                    oc,
                    sial,
                )
        elif (
            int(args.num_of_struct) > 1
            and int(args.grow_h) == 0
            and (args.rxn_type == "ht")
        ):
            O_name_index, closest_O_index, h_loc, sial, oc = closest(
                un_atom_names,
                int(args.num_of_struct),
                com_data[i],
                wcom_data[i],
                O_coor,
                wO_coor,
                si_coor,
                lattice,
                float(args.cut_off),
                counter_array,
                com_counter_array,
                args.inp1_file,
                args.inp3_file,
                n1[i],
                float(args.ref_angle),
                float(args.ref_dis),
                hydrogen_coor_data[i],
                int(args.grow_h),
                args.rxn_type,
                args.zeo_name,
            )
            if O_name_index != -1:
                final_coor = np.array(
                    [
                        [float(item) for item in whole_line.split()[1:4]]
                        for whole_line in readf[
                            (i) * (int(args.mol_size) + start)
                            + start : (i + 1) * (int(args.mol_size) + start)
                        ]
                    ]
                )
                grow_hydrogen_to_reactant_state_ht(
                    "osite_info.xyz",
                    lattice,
                    final_coor[: len(final_coor) - 3],
                    elements[: len(elements) - 3],
                    counter_array[O_name_index],
                    un_atom_names[O_name_index],
                    float(args.cut_off),
                    int(args.grow_h),
                    oc,
                    sial,
                )
        else:
            O_name_index, closest_O_index, h_loc, sial, oc = closest(
                un_atom_names,
                int(args.num_of_struct),
                com_data[i],
                wcom_data[i],
                O_coor,
                wO_coor,
                si_coor,
                lattice,
                float(args.cut_off),
                counter_array,
                com_counter_array,
                args.inp1_file,
                args.inp3_file,
                n1[i],
                float(args.ref_angle),
                float(args.ref_dis),
                hydrogen_coor_data[i],
                int(args.grow_h),
                args.rxn_type,
                args.zeo_name,
            )
            if O_name_index != -1:
                final_coor = np.array(
                    [
                        [float(item) for item in whole_line.split()[1:4]]
                        for whole_line in readf[
                            (i) * (int(args.mol_size) + start)
                            + start : (i + 1) * (int(args.mol_size) + start)
                        ]
                    ]
                )
                print("structure found at i =")
                print(i)
                print("before retransform h location")
                print(h_loc)
                print("before retransform si to al location")
                print(sial)
                h_loc = retransform(np.array(h_loc).reshape(-1, 3), lattice)
                sial = retransform(np.array(sial).reshape(-1, 3), lattice)
                print("after retransform h location")
                print(h_loc)
                print("after retransform si to al location")
                print(sial)
                create_cif_file(lattice, final_coor)
                create_xyz_with_added_hydrogens("temp_cif.cif")

                openf1 = open("st_with_H.xyz")
                readf1 = openf1.readlines()
                linecount = len(readf1)
                elements = extract_atom_indices(readf1[start:linecount])
                print(elements)

                read = np.loadtxt(fname="st_with_H.xyz", skiprows=2, usecols=(1, 2, 3))
                tr_coor = retransform(read, lattice)
                mod_tr_coor = np.concatenate((tr_coor, h_loc.reshape(-1, 3)), axis=0)
                create_final_pdb_with_zeolite(
                    mod_tr_coor,
                    sial,
                    elements,
                    lattice,
                    "osite_info.xyz",
                    un_atom_names[O_name_index],
                    float(args.cut_off),
                    counter_array[O_name_index],
                    energies[i],
                )
        if all(p == int(args.num_of_struct) for p in counter_array):
            break
