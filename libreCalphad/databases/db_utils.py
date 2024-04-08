from importlib import resources as impresources
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