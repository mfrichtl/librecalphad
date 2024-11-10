# About
Welcome to the LibreCalphad project. The purpose is to create a set of open CALPHAD databases and property models for use with open-source modeling software, such as [pycalphad](https://pycalphad.org/docs/latest/#). It is currently (and probably always will be) a work in progress. User discretion is advised. I make no assertions as to the accuracy of the results and highly recommend you validate them versus other CALPHAD software or reliable references.  If you want to help make this software better, please do. See the contributing section below for more information.

# Steel Database
- The steel database file (mf-steel.tdb) is decent, but there are too many opportunities for improvement to list. Known errors are documented in the frontmatter, though other issues probably exist.
- The mobility and molar volume databases (mf-mobility.tdb and mf-volume.tdb) are designed to be used in conjunction with mf-steel.tdb. Therefore, things like phase definitions are not contained in these support databases, which will likely cause errors if you try to use them standalone.

The rest of the files don't really do much for now. More will be added to this at a later date.

# Property Models
A martensite-start temperature model for steels is currently available. It still needs much work, but is a good start for collaboration. See the README in it's folder or the notebook in the examples folder for more information.

# Contributing
Do you want to help improve this software? Great! Unsolicited pull requests are gladly appreciated! If you find an issue with a system, but do not know how to help fix it, open an issue and provide as much information as you can. Images of current behavior and references to expected behavior will help tremendously. Please check that this particular issue isn't documented elsewhere.
