Database files go here. For now only a steels database is provided, but hopefully more will come.

# mf-steel
This is my steels database. I know it needs work. Please help! mf-steel-3g is intended to be a 3rd-generation CALPHAD database, but is not anywhere close to as comprehensive as mf-steel.tdb. There are some challenges with implementing a 3rd-generation CALPHAD database in pycalphad for the time being, but I'm sure it will improve with time. Please be patient.

If you're just interested in trying to use it, see the python files in the validation folder for numerous examples on how to generate binary and ternary phase diagrams. Many commercial softwares disable certain phases automatically, but this will not. If you see some strange phase being predicted, particularly gas or an ordered phase (pycalphad does not always handle ordering correctly), try disabling it first by following the examples in the python files. If it's something that otherwise seems totally wrong, feel free to open an issue. I will respond and try to help. Just be aware that a lot of the thermodynamic assessments for steels are old and are not fully compatible with new work, which can create some strange issues when trying to incorporate them.

Validation files are contained in the mf-steel_validation folder and are provided in subfolders for the respective systems. The phase diagrams contained therein were all generated using mf-steel.tdb. I've done my best to document issues in the frontmatter in the database files themselves, but I'm sure there are plenty more that I'm not yet aware of.

# Contributing
Do you want to contribute? Great! I love pull requests! If you have assembled a database file that you don't mind sharing, that's amazing! If you know of better thermodynamic assessments than what I'm currently using or for systems that aren't currently incorporated, feel free to let me know.

I'm also interested in other ideas to improve the state of this project. Constructive criticism is also greatly appreciated!