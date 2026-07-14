""""
Functions for processing raw high frequency data into computed fluxes.

Includes modules for
- Despiking+cleaning data (`clean`)
- Axis rotation (`rotate`)
- Spectral analysis + correction (`spectral`)
- Time lag compensation (`timelag`)
- Miscellaneous instrument specific corrections (`instrument`) such as transducer shadowing corrections (e.g. Kaimal corrections), Gill Windmaster w-boost corrections, LI-7500 self-heating issues (e.g. Burba Correction), and other instrument-specific issues.  
- Actual flux computation (`compute`)
- Other utility functions (`utils`)
"""