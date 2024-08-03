from importlib import resources as impresources
import itertools
from io import StringIO
from pycalphad import Database


def load_database(files):
    """
    Load a database file(s) from the databases directory and return a pycalphad Database object.

    Parameters : file, string or list
                    Name of the database file(s) to load.

    Returns : db, pycalphad Database
                The pycalphad Database object from the database file.
    """

    if type(files) == str:
        dbf = str(impresources.files('libreCalphad.databases') / files)
    elif type(files) == list:
        dbf = StringIO()
        for file in files:
            impfile = impresources.files('libreCalphad.databases') / file
            with open(impfile, 'r') as f:
                for line in f.readlines():
                    dbf.write(line)
        dbf = dbf.getvalue()
    return Database(dbf)


def check_endmembers(db, phase, ref='!'):
    """
    Function to check for missing endmembers and initialize them with a simple sum of SER terms for each component.
    TODO: Allow for automatic testing of systems with new endmembers.

    Parameters : db, string
                    A string pointing to the thermodynamic database to check.
                 phase, string
                    The phase to check. Must match a phase definition line from the database.
                 ref, string
                    The reference to append to the end of each new line.

    Returns : out_string, string
                A string of the endmember definitions for the phase.
              missing_endmembers, list
                A list of the new endmembers that were created. Useful for testing systems for compatibility.
    """
    
    constituent_dict = {}
    site_dict = {}
    current_endmembers = {}
    current_line = ''
    complete_line = False
    out_string = ''

    with open(db) as f:
        for line in f.readlines():
            line = line.rstrip()
            if line.lstrip().startswith('$'):  # comments and blank lines should not be processed
                current_line = ''
                continue
            if line.endswith("'"):  # multi-line database info or reference
                current_line = ''
                continue
            if line == '':
                continue
            if line.endswith('!'):
                if complete_line:  # Should be a single line parameter
                    current_line = line
                else:  # should be a mult-line parameter
                    current_line += line
                complete_line = True
            else:  # not a complete line
                if complete_line:  # should be the first line of a multi-line parameter
                    current_line = line
                else:  # Should be a middle line  of a multi-line parameter
                    current_line += line
                current_line += '\n'
                complete_line = False
                continue  # go to next line
            
            # Need to make sure I understand the phase definition first
            if current_line.startswith(f'PHASE {phase}'):
                splitted_line = current_line.split(' ')
                sl = 0
                i = 0
                for entry in splitted_line:
                    try:
                        entry = int(entry)
                        if i == 0:  # assume first integer is the number of sublattices
                            num_sublattices = entry
                        else:
                            site_dict[sl] = entry
                            sl += 1
                        i += 1
                    except:
                        continue
            # Next build the constituents on each sublattice
            if current_line.startswith(f'CONSTITUENT {phase}'):  # get the constituent definitions
                splitted_line = current_line.replace(' ', '').strip().split(':')
                sl = len(constituent_dict.keys())
                for i in splitted_line:
                    i = i.strip('\n')
                    if len(i.split(',')) > 1:
                        constituent_dict[sl] = i.strip('\n').split(',')
                        sl += 1
                    elif len(i) <= 2 and i != '!':  # single component sublattice
                        constituent_dict[sl] = [i]  # brackets needed for compatibility with product function below
                        sl += 1

            # Now figure out which endmembers are present
            if current_line.startswith(f'PARAMETER G({phase}'):
                endmember = current_line.split(',')[1].split(';')[0]
                # Check endmember definition formatting
                assert len(endmember.split(' ')) == 1, f"Improperly formatted endmember -- {endmember}"
                # Check to make sure this is a valid endmember
                for i in range(len(endmember.split(':'))):
                    component = endmember.split(':')[i]
                    assert component in constituent_dict[i], f"Extra endmember identified, {endmember} \n {current_line} \n {constituent_dict[i]}"
                current_endmembers[endmember] = current_line

    # Next determine what endmembers need to be present
    required_endmembers = list(itertools.product(*list(constituent_dict.values())))
    missing_endmembers = []
    insert_blank_line = False
    current_group_component = ''  # key on the second from the last sublattice for grouping
    doubled_components = ['B', 'C', 'H', 'N', 'O', 'P', 'S']  # these components have a GHSERXX function definition for their SER
    for endmember in required_endmembers:
        if current_group_component != endmember[num_sublattices-2]:
            insert_blank_line = True
            current_group_component = endmember[num_sublattices-2]
        else:
            insert_blank_line = False
        if insert_blank_line:
            out_string += '\n'
        endmember = ':'.join(endmember)
        if endmember not in list(current_endmembers.keys()):
            out_string += f"PARAMETER G({phase},{endmember};0) 298.15\n"
            missing_endmembers.append(endmember)
            next_line = ' '
            for i in site_dict.keys():
                component = endmember.split(':')[i]
                
                if component == 'VA':
                    continue
                if component in doubled_components:
                    component = component + component
                if site_dict[i] == 1:
                    next_line += f" +GHSER{component}#"
                else:
                    next_line += f" +{site_dict[i]}*GHSER{component}#"
            next_line += ';  6000 N'
            while len(next_line) < 72:
                next_line += ' '
            next_line += ref + '\n'
            out_string += next_line
        else:
            out_string += current_endmembers[endmember]
            out_string += '\n'
    
    return out_string, missing_endmembers

def find_missing_functions(db_file):
    """
    Function to find missing function definitions in a database file. This is particularly useful for catching typos.

    Parameters : db_file, string
                String pointer to the database file.
    
    Returns : missing_funcs, list
                A list of the missing function definitions.
    """
    func_defs = []
    duplicate_funcs = []
    missing_funcs = []

    with open(db_file) as f:
        for line in f.readlines():
            if 'FUNCTION' in line:  # function definitions
                this_func = line.split(' ')[1]
                if this_func in func_defs:
                    duplicate_funcs.append(this_func)
                else:
                    func_defs.append(this_func)

    with open(db_file) as f:
        for line in f.readlines():
            for term in line.split(' '):
                if term.endswith('#'):
                    split_term = term[1:].split('*')
                    if len(split_term) > 1:
                        func = split_term[-1][:-1]
                    else:
                        func = split_term[0][:-1]
                    if func not in func_defs and func not in missing_funcs:
                        missing_funcs.append(func)
    return missing_funcs