allocation_construct.py
========================

Python module to perform life cycle assessment (LCA)  allocations and input-output (IO) constructs from supply and use inventory tables (SUT).

Each function accepts a table describing the use of products by industries and a table of their emissions and resource use. It then allocates/constructs these values based on the supply flows of each industry.

These functions can process use of products that are traceable to specific providers (traceable use) or average commodities from homogeneous markets (untraceable use).


Models
------

- Partition allocations and constructs (PA/PC)
	- including industry technology construct (ITC)
	- including "surplus method" and European systems construct (ESC)

- Product substitution allocations and constructs (PSA/PSC)
	- including byproduct technology construct (BTC)

- Alternate activity allocations and constructs (AAA/AAC)
	- including commodity technology construct (CTC)


Documentation
-------------

This module follows closely the notation and equations of the following articles. Please cite.

> Majeau-Bettez, G., R. Wood, and A.H. Strømman. 2014. Unified Theory of Allocations and Constructs in Life Cycle Assessment and Input-Output Analysis. *Journal of Industrial Ecology* 18(5): 747–770. [10.1111/jiec.12142](http://dx.doi.org/10.1111/jiec.12142)

> Majeau-Bettez, G., R. Wood, E.G. Hertwich, and A.H. Strømman. 2014. When do allocations and constructs respect material, energy, financial, and production balances in LCA and EEIO? *Journal of Industrial Ecology* In Press: In Press.

The variables are described succintly in readMe\_Variables.txt


Dependencies
------------

- Python 3
- numpy
