-------------------------------------------------------------------------------------
 	VAR	DIMENSION		DESCRIPTION
-------------------------------------------------------------------------------------

NOTATION:
  	com	integer			number of commodities
  	ind	integer			number of activities
  	org	integer			number of activities of origin
	ext	integer			number of (environmental) extensions
	THETA	integer			Industrustry wildcard
	theta	integer			commodity wildcard
	ORG	integer			Industry of origin wildcard

OPERATORS:
	E_bar 	[com, ind]		Primary product of each industry
  	e_com	[com,1]			vertical vector of ones
  	e_ind	[ind,1]			vertical vector of ones
  	nn_in	[com]			filter, boolean: true = row not null
  	nn_out	[com]			filter, boolean: true = column not null

INVENTORY DATA:
 	U/Uu 	[com, ind]		untraceable Use table
  or	U/Ut	[ind,com,ind]		traceable Use table
 	V 	[com, ind]		Supply table
  	G	[ext, ind]		Environmental extension inventory
	h	[com,1]			Inventoried final consumption

INTERMEDIATE, CALCULATED VARIABLES:
  	g	[1,ind]			total output of each industry
  	q	[com,1]			total production of each commodity
	B	[com,ind]		Use coefficients
	C	[com,ind]		Product Mix
	D	[ind,com]		Market share
	M	[com,ind]		secondary product per unit of primary production
	N	[com,ind]		Industry requirements per unit of primary production

INTERMEDIATE, ASSUMED VARIABLE:
	DeltV	[com,ind]		Alterations to the inventoried supply (V)
	DeltU	[com,ind]		Alteration to the inventoried untraceable Use
  or 	DeltU	[org,com,ind]		Alteration to the inventoried traceable Use
	PSI	[com,ind]		Intrinsic property of coproducts
        psi_ind	[1,ind]			Activity-wide unique intrinsic property
	PHI	[ind,com]		Partition coefficients
	Gamma 	[ind, com]		Alternate Activity for each product
	A_gamma	[same as Z]		Alternate technology coefficient matrix
	Theta 	[ind, com]		Competing industry for each product
  	Xi	[com, com]		Substitution Matrix

FINAL VARIABLES:
	y	[com,1]			Exogeneously defined final demand
  	Z	[com,ind,com]		asymmetric Intermediate flow matrix
  or	Z	[ind,com,ind,com]	symmetric traceable Intermediate flow matrix
  or	Z	[com,com]		symmetric aggregation Intermediate flow matrix
  	A	[same as Z]		Normalized technical requirements
	x	[com,1]			Calculated production levels
  	G_all	[ext,ind,com]		allocated  factors of production
  or	G_con	[ext,com]		allocated  factors of production
  	F	[ext,ind,com]		Normalizied factors of production
  or	F	[ext,com]		Normalizied factors of production
-------------------------------------------------------------------------------------
