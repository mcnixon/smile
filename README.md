# smile
Structural Model of Internal Layers of Exoplanets

Installation

Once you have cloned the repository, you will need to download the equation of state (EOS) data and define an environment variable setting the path to the data files.

These data can be accessed at the following link:

https://umd.box.com/s/ol5z2np499shiukpru2y6cl2sm1x85vm

Once you have downloaded everything, you need to create an environment variable pointing to the path of the file. At the command line, enter

export eos_path="your/path/to/eos/data"

Where you replace the text inside the "" with the path to your EOS data files. To avoid having to repeat this step, you can add the above line to your .zshrc or .bashrc file, depending on your OS.
