# -*- coding: utf-8 -*-
"""
Perform allocations and constructs on LCA or EEIO Supply and Use Inventories

Collection of functions that perform allocation or construct modelling on
inventories recorded as Supply and Use Tables (SUT). These functions are
intended for use in Lifecycle Assessment (LCA) and Environmentally Extended
Input-Output studies (EEIO).

It can accommodate untraceable accounts (SuUT) or traceable ones (StUT) as well
as product flows and environmental extensions.

Functions can perform partitioning (PA), alternate activity allocation (AAA),
product substitution allocation (PSA) and lump-sum allocation (LSA), along with
their associated aggregation constructs (PC, AAC, PSC and LSC). This module's
functions can also perform special cases of these constructs, namely Industry
Technology, Commodity Technology and Byproduct Technology Constructs.

This module was written to demonstrate and illustrate equations in:

    Majeau-Bettez, G., R. Wood, and A.H. Strømman. 2014. Unified Theory of
    Allocations and Constructs in Life Cycle Assessment and Input-Output
    Analysis. Journal of Industrial Ecology 18(5): 747–770.
    DOI:10.1111/jiec.12142

Close correspondence to the original equations was favoured over computational
speed or simplicity of notation.

Uppercase variables indicate an array with at least two dimensions. Lowercase
letters represent vectors

:copyright: 2013, Guillaume Majeau-Bettez, Richard Wood, Anders H. Strømman
:license: BSD 2-Clause License

"""
import numpy as np
# pylint: disable-msg=C0103


##############################################################################
# PARTITION

def pa_coeff(V, PSI):
    """Calculates partition coefficients from supply and properties table

    Parameters
    ----------
    V : Supply table [com, ind]
    PSI : Properties table [com, properties]

    Returns
    --------
    PHI : Partition coefficient [ind, com (default=np.empty(0))]
        Properties in PSI should be intensive properties (e.g. energy density,
        price etc., not extensive properties such as energy content, value, or
        mass

    """
    # Calculate total amount of the partition property that is output by each
    # industry (total mass output for all commodities supplied by ind. J)
    denominator = ddiag(V.T.dot(PSI))

    # Calculate the share of this total output of property that is mediated by
    # each output (share of total mass output by ind. J that happens via
    # commodity j.
    PHI = np.linalg.inv(denominator).dot(V.T * PSI.T)  # <-- eq:PCHadamard
    return PHI


def pa(U, V, PSI, PHI=np.empty(0), G=np.empty(0)):
    """Performs Partition Allocation of a SuUT or StUT inventory

    Parameters
    ----------
    U : Use table [com, ind] or [org, com, ind]
    V : Supply table [com, ind]
    PSI : Properties table [com, properties]
    PHI : Partition coefficient [ind, com (default=np.empty(0))]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    Z : allocated intermediate flow matrix [com,ind,com] ¦ [org,com,ind,com]
    A : Normalized technical requirements (2-dimensional)
    nn_in : filter to remove np.empty rows in A or Z [com] | [org*com]
    nn_out : filter to remove np.empty columns in A or Z [ind*com]
    G_all : Allocated emissions [ext,ind,com]
    F : Normalized, allocated emissions [ext, com*ind]
    """
    # default values:
    G_all = np.empty(0)
    F = np.empty(0)

    # Basic variables
    (com, ind, org, traceable, _, _, _, _, ext) = basic_variables(U, V, G)

    # Partitioning properties and coefficients
    # N.B. Improve control: if both PHI and PSI np.empty...
    if not PHI.size:
        PHI = pa_coeff(V, PSI)

    # Partitioning of product flows
    if not traceable:
        Z = np.zeros((com, ind, com))
        for J in range(ind):
            # eq:partition_allocation
            Z[:, J, :] = np.outer(U[:, J], PHI[J, :])
    else:
        Z = np.zeros((org, com, ind, com))
        for I in range(org):
            for J in range(ind):
                #eq:PAtrace
                Z[I, :, J, :] = np.outer(U[I, :, J], PHI[J, :])

    # Normalize system description
    (A, nn_in, nn_out) = matrix_norm(Z, V)

    # Partitioning of environmental extensions
    if G.size:
        G_all = np.zeros((ext, ind, com))
        for J in range(ind):
            G_all[:, J, :] = np.outer(G[:, J], PHI[J, :])

        # Normalize environmental extension
        (F, _, _) = matrix_norm(G_all, V)

    return (Z, A, nn_in, nn_out, G_all, F)


def pc_agg(U, V, PSI, PHI=np.empty(0), G=np.empty(0)):
    """Performs Partition Aggregation Construct of SuUT inventory

    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    PSI : Properties table [com, properties]
    PHI : Partition coefficient [ind, com (default=np.empty(0))]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    Z : constructed intermediate flow matrix [com,com]
    A : Normalized technical requirements [com,com]
    nn_in : filter to remove np.empty rows in A or Z [com]
    nn_out : filter to remove np.empty columns in A or Z [com]
    G_con : Constructed emissions [ext,com]
    F : Normalized, constructed emissions [ext, com]

    """
    # default values:
    G_con = np.empty(0)
    F = np.empty(0)

    # Partitioning properties and coefficients
    if not PHI.size:
        PHI = pa_coeff(V, PSI)

    # Partitioning of product flows
    Z = U.dot(PHI)  # <-- eq:PCagg
    (A, nn_in, nn_out) = matrix_norm(Z, V)

    # Partitioning of environmental extensions
    if G.size:
        G_con = G.dot(PHI)  # <-- eq:PCEnvExt

        # Normalize environmental extension
        (F, _, _) = matrix_norm(G_con, V)

    return (Z, A, nn_in, nn_out, G_con, F)


##############################################################################
def psa(U, V, E_bar, Xi, Theta=np.empty(0), G=np.empty(0)):
    """Performs Product Substitution Allocation of SuUT or StUT inventory

    Parameters
    ----------
    U : Use table [com, ind] or [org, com, ind]
    V : Supply table [com, ind]
    E_bar : 0 or 1 mapping of primary commodities to industries [com,ind]
    Xi : substitution table [com,com]
    Theta : Identifies activity against which secondary production competes
           [ind,com]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    Z : allocated intermediate flow matrix [com,ind,com] ¦ [org,com,ind,com]
    DeltV : modelled change in supply [com,ind]
    A : Normalized technical requirements (2-dimensional)
    nn_in : filter to remove np.empty rows in A or Z [com] | [org*com]
    nn_out : filter to remove np.empty columns in A or Z [ind*com]
    G_all : Allocated emissions [ext,ind,com]
    F : Normalized, allocated emissions [ext, com*ind]

    """
    # Default values
    G_all = np.empty(0)
    F = np.empty(0)

    # Basic variables
    (com, ind, org, traceable, _, e_ind, _, _, ext) = basic_variables(
            U, V, G)
    (V_tild, V_bar, U_tild, _) = _rank_products(E_bar, V, U)
    DeltV = V_tild

    # Allocation of Product Flows
    if not traceable:
        Z = np.zeros((com, ind, com))
        for J in range(ind):
            # eq:PSAUntrace
            sFlows = U[:, J] - Xi.dot(V_tild[:, J])
            Z[:, J, :] = np.outer(sFlows, E_bar[:, J].T)
    else:
        U_bar = U - U_tild
        Z = np.zeros((org, com, ind, com))
        for J in range(ind):
            for I in range(org):
                #eq:PSAtrace
                Z[I, :, J, :] = np.outer(
                        U_bar[I, :, J]
                        - Theta[I, :].dot(ddiag(Xi.dot(V_tild[:, J])))
                        + Theta[I, :].dot(ddiag(e_ind.dot(U_tild[:, :, J]))),
                   E_bar[:, J].T)
    # Normalizing
    (A, nn_in, nn_out) = matrix_norm(Z, V_bar)

    # Allocation of Environmental Extensions
    if G.size:
        G_all = np.zeros((ext, ind, com))
        for J in range(ind):
            # eq:PSAEnvExt
            G_all[:, J, :] = np.outer(G[:, J], E_bar[:, J].T)
        # Normalization
        (F, _, _) = matrix_norm(G_all, V_bar)

    # Return allocated values
    return(Z, DeltV, A, nn_in, nn_out, G_all, F)


def psc_agg(U, V, E_bar, Xi, G=np.empty(0)):
    """Performs Product Substitution aggregation Construct of SuUT inventory

    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    E_bar : 0 or 1 mapping of primary commodities to industries [com,ind]
    Xi : substitution table [com,com]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    Z : constructed intermediate flow matrix [com,com]
    A : Normalized technical requirements [com,com]
    nn_in : filter to remove np.empty rows in A or Z [com]
    nn_out : filter to remove np.empty columns in A or Z [com]
    G_con : Constructed emissions [ext,com]
    F : Normalized, constructed emissions [ext, com]
    """

    # Default values
    G_con = np.empty(0)
    F = np.empty(0)

    # Basic variables
    (V_tild, V_bar, _, _) = _rank_products(E_bar, V)

    # Construction of Product Flows
    Z = (U - Xi.dot(V_tild)).dot(E_bar.T)  # <-- eq:PSCagg
    # Normalizing
    (A, nn_in, nn_out) = matrix_norm(Z, V_bar)

    # Allocation of Environmental Extensions
    if G.size:
        G_con = G.dot(E_bar.T)  # <-- eq:NonProdBalEnvExt

        # Normalization
        (F, _, _) = matrix_norm(G_con, V_bar)

    # Return allocated values
    return(Z, A, nn_in, nn_out, G_con, F)


##############################################################################

def alternate_tech(U, V, E_bar, Gamma, nmax=np.Inf, lay=None):
    """Compilation of Alternate Technologies for use in AAA and AAC models

    Parameters
    ----------
    U : Use table [com, ind] or [org, com, ind]
    V : Supply table [com, ind]
    E_bar : mapping of primary commodities to industries [com,ind]
    Gamma : mapping of alternate producer for each commodity [ind,com]
    nmax : maximum number of iterations, as this search for alternative
       technologies is not garanteed to suceed

    Returns
    --------
    A_gamma : the selected alternative technology that will be assumed for
        each secondary production


    """

    # Basic variables
    (com, _, org, traceable, e_com, _, _, g, _) = basic_variables(U, V)
    #
    (V_tild, V_bar, _, _) = _rank_products(E_bar, V)

    # If a property layer is defined for Gamma, then evaluate unit conversion
    if Gamma.ndim == 3:
        if lay is None:
            raise TypeError('expected a value for lay')
        s = Gamma.shape
        tmp = np.zeros((s[0], s[2]))
        for i in range(Gamma.shape[1]):
            tmp += diaginv(lay[i, :].dot(E_bar)).dot(Gamma[:, i, :]).dot(
                    ddiag(lay[i, :]))
        Gamma = tmp

    so = np.array(np.sum(V != 0, 0) == 1, dtype=int)
    mo = np.array(np.sum(V != 0, 0) != 1, dtype=int)

    invg = diaginv(g)
    M = V_tild.dot(np.linalg.inv(ddiag(e_com.dot(V_bar))))
    # Prepare summation term used in definition of A_gamma
    n = 0
    tier = -1 * Gamma.dot(M)
    tier_n = np.linalg.matrix_power(tier, 0)   # simplifies to identity matrix
    theSum = tier_n.dot(Gamma)
    n = n + 1
    while np.sum(tier_n) != 0 and n <= nmax:
        tier_n = tier_n.dot(tier)
        theSum = theSum + tier_n.dot(Gamma)
        n += 1
    if not traceable:
        B = U.dot(invg)
        B_so = B.dot(ddiag(so))

        N = U.dot(np.linalg.inv(ddiag(e_com.dot(V_bar))))
        N_so = N.dot(ddiag(mo))

        A_gamma = (B_so + N_so).dot(theSum)

    else:
        A_gamma = np.zeros([org, com, com])
        for I in range(org):
            Bo = U[I, :, :].dot(invg)
            Bo_so = Bo.dot(ddiag(so))
            No = U[I, :, :].dot(np.linalg.inv(ddiag(e_com.dot(V_bar))))
            No_mo = No.dot(ddiag(mo))
            A_gamma[I, :, :] = (Bo_so + No_mo).dot(theSum)

    return(A_gamma)


def aaa(U, V, E_bar, Gamma, G=np.empty(0), nmax=np.Inf, lay=None):
    """ Alternate Activity Allocation of StUT or SuUT inventory

    Parameters
    ----------
    U : Use table [com, ind] or [org, com, ind]
    V : Supply table [com, ind]
    E_bar : 0 or 1 mapping of primary commodities to industries [com,ind]
    Gamma : 0 or 1 mapping of alternate activity for each commodity
       [ind,com]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))
    nmax : maximum number of iterative loops for defining A_gamma
       (default=Inf)

    Returns
    --------
    Z : allocated intermediate flow matrix [com,ind,com] ¦ [org,com,ind,com]
    A : Normalized technical requirements (2-dimensional)
    nn_in : filter to remove np.empty rows in A or Z [com] | [org*com]
    nn_out : filter to remove np.empty columns in A or Z [ind*com]
    G_all : Allocated emissions [ext,ind,com]
    F : Normalized, allocated emissions [ext, com*ind]

    """
    # Default outputs
    G_all = np.empty(0)
    F = np.empty(0)

    # Basic variables
    (com, ind, org, traceable, _, _, _, _, ext) = basic_variables(U, V, G)
    (V_tild, _, _, _) = _rank_products(E_bar, V)

    # Calculate competing technology requirements
    A_gamma = alternate_tech(U, V, E_bar, Gamma, nmax, lay)

    if G.size:
        F_gamma = alternate_tech(G, V, E_bar, Gamma, nmax, lay)

    # Allocation step
    if not traceable:
        Z = np.zeros((com, ind, com))
        for J in range(ind):
            # eq:aaa
            Z[:, J, :] = np.outer(U[:, J] - A_gamma.dot(V_tild[:, J]),
                    E_bar[:, J].T) + A_gamma.dot(ddiag(V_tild[:, J]))
    else:
        Z = np.zeros((org, com, ind, com))
        for I in range(org):
            for J in range(ind):
                #eq:AAAtrace
                Z[I, :, J, :] = np.outer(
                        U[I, :, J] - A_gamma[I, :, :].dot(V_tild[:, J]),
                        E_bar[:, J].T
                        ) + A_gamma[I, :, :].dot(ddiag(V_tild[:, J]))
    (A, nn_in, nn_out) = matrix_norm(Z, V)

    # Partitioning of environmental extensions
    if G.size:
        G_all = np.zeros((ext, ind, com))
        for J in range(ind):
            #eq:AAAEnvExt
            G_all[:, J, :] = np.outer(
                    G[:, J] - F_gamma.dot(V_tild[:, J]), E_bar[:, J].T) + \
                    F_gamma.dot(ddiag(V_tild[:, J]))
        (F, _, _) = matrix_norm(G_all, V)

#   # Output
    return(Z, A, nn_in, nn_out, G_all, F)


def aac_agg(U, V, E_bar, Gamma, G=np.empty(0), nmax=np.Inf):
    """ Alternative Activity aggregation Construct of SuUT inventory

    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    E_bar : 0 or 1 mapping of primary commodities to industries [com,ind]
    Gamma : 0 or 1 mapping of alternate activity for each commodity
           [ind,com]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))
    nmax : maximum number of iterative loops for defining A_gamma
           (default=Inf)

    Returns
    --------
    Z : constructed intermediate flow matrix [com,com]
    A : Normalized technical requirements [com,com]
    nn_in : filter to remove np.empty rows in A or Z [com]
    nn_out : filter to remove np.empty columns in A or Z [com]
    G_con : Constructed emissions [ext,com]
    F : Normalized, constructed emissions [ext, com]

    """
    # Default outputs
    G_con = np.empty(0)
    F = np.empty(0)

    # Basic variables
    (_, _, _, _, _, e_ind, _, _, _) = basic_variables(U, V, G)
    (V_tild, _, _, _) = _rank_products(E_bar, V)

    # Calculate competing technology requirements
    A_gamma = alternate_tech(U, V, E_bar, Gamma, nmax)

    # Allocation step
    Z = (U - A_gamma.dot(V_tild)).dot(E_bar.T) + \
            A_gamma.dot(ddiag(V_tild.dot(e_ind)))  # <-- eq:AACagg
    (A, nn_in, nn_out) = matrix_norm(Z, V)

    # Partitioning of environmental extensions
    if G.size:
        F_gamma = alternate_tech(G, V, E_bar, Gamma, nmax)
        G_con = (G - F_gamma.dot(V_tild)).dot(E_bar.T) + \
                F_gamma.dot(ddiag(V_tild.dot(e_ind)))  # <-- eq:AACEnvExt
        (F, _, _) = matrix_norm(G_con, V)

#   Output
    return(Z, A, nn_in, nn_out, G_con, F)

##############################################################################


def lsa(U, V, E_bar, G=np.empty(0)):
    """ Performs Lump-sum Allocation of SuUT Inventory

    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    E_bar : 0 or 1 mapping of primary commodities to industries [com,ind]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    Z : allocated intermediate flow matrix [com,ind,com]
    DeltV : modelled change in supply [com,ind]
    A : Normalized technical requirements (2-dimensional)
    nn_in : filter to remove np.empty rows in A or Z [com]
    nn_out : filter to remove np.empty columns in A or Z [ind*com]
    G_all : Allocated emissions [ext,ind,com]
    F : Normalized, allocated emissions [ext, com*ind]

    N.B. This model is not defined for traceable SUT inventory
    """
    # Default values
    G_all = np.empty(0)
    F = np.empty(0)

    # Basic variables
    (com, ind, _, _, e_com, _, _, g, ext) = basic_variables(U, V, G)
    (V_tild, _, _, _) = _rank_products(E_bar, V)
    Z = np.zeros((com, ind, com))
    V_dd = np.zeros(V.shape)

    # Allocation of Product Flows

    DeltV = V_tild - E_bar.dot(ddiag(e_com.T.dot(V_tild)))

    for J in range(ind):
        Z[:, J, :] = np.outer(U[:, J], E_bar[:, J].T)  # <-- eq:LSA
        V_dd[:, J] = E_bar[:, J].dot(g[J])  # <-- eq:LSA

    # Normalizing
    (A, nn_in, nn_out) = matrix_norm(Z, V_dd)

    # Allocation of Environmental Extensions
    if G.size:
        G_all = np.zeros((ext, ind, com))
        for J in range(ind):
            # eq:LSAEnvExt
            G_all[:, J, :] = np.outer(G[:, J], E_bar[:, J].T)

        # Normalization
        (F, _, _) = matrix_norm(G_all, V_dd)

    # Return allocated values
    return(Z, DeltV, A, nn_in, nn_out, G_all, F)


def lsc(U, V, E_bar, G=np.empty(0)):
    """ Performs Lump-sum aggregation Construct of SuUT inventory

    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    E_bar : 0 or 1 mapping of primary commodities to industries [com,ind]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))


    Returns
    --------
    Z : constructed intermediate flow matrix [com,com]
    A : Normalized technical requirements [com,com]
    nn_in : filter to remove np.empty rows in A or Z [com]
    nn_out : filter to remove np.empty columns in A or Z [com]
    G_con : Constructed emissions [ext,com]
    F : Normalized, constructed emissions [ext, com]

    """
    # Default values
    G_con = np.empty(0)
    F = np.empty(0)
    (_, _, _, _, _, _, _, g, _) = basic_variables(U, V)

    # Allocation of Product Flows
    Z = U.dot(E_bar.T)  # <-- eq:LSCagg
    V_dd = E_bar.dot(ddiag(g))  # <-- eq:LSCagg
    # Normalizing
    (A, nn_in, nn_out) = matrix_norm(Z, V_dd)

    # Allocation of Environmental Extensions
    if G.size:
        G_con = G.dot(E_bar.T)  # <-- eq:NonProdBalEnvExt
        # Normalization
        (F, _, _) = matrix_norm(G_con, V_dd)

    # Return allocated values
    return(Z, A, nn_in, nn_out, G_con, F)


###############################################################################
# SPECIAL CASES

def itc(U, V, G=np.empty(0)):
    """Performs Industry Technology Construct of SuUT inventory

    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    Z : constructed intermediate flow matrix [com,com]
    A : Normalized technical requirements [com,com]
    G_con : Constructed emissions [ext,com]
    F : Normalized, constructed emissions [ext, com]

    """
    # Default output
    G_con = np.empty(0)
    F = np.empty(0)
    # Basic Variables
    (_, _, _, _, _, _, _, g, _) = basic_variables(U, V, G)

    Z = U.dot(diaginv(g)).dot(V.T)  # <-- eq:itc
    (A, _, _) = matrix_norm(Z, V)

    if G.size:
        G_con = G.dot(diaginv(g)).dot(V.T)  # <-- eq:ITCEnvExt
        (F, _, _) = matrix_norm(G_con, V)

    return(Z, A, G_con, F)


def ctc(U, V, G=np.empty(0)):
    """Performs Commodity Technology Construct of SuUT inventory

    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    Z : constructed intermediate flow matrix [com,com]
    A : Normalized technical requirements [com,com]
    G_con : Constructed emissions [ext,com]
    F : Normalized, constructed emissions [ext, com]

    """

    # Default output
    G_con = np.empty(0)
    F = np.empty(0)
    # Basic Variables
    (_, _, _, _, _, _, q, _, _) = basic_variables(U, V, G)
    A = U.dot(np.linalg.inv(V))  # <-- eq:ctc
    Z = A.dot(ddiag(q))

    if G.size:
        F = G.dot(np.linalg.inv(V))
        G_con = F.dot(ddiag(q))  # <--eq:CTCEnvExt
    return(Z, A, G_con, F)


def btc(U, V, E_bar=np.empty(0), G=np.empty(0)):
    """Performs Byproduct Technology Construct of SuUT inventory
    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    E_bar : 0 or 1 mapping of primary commodities to industries [com,ind]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    Z : constructed intermediate flow matrix [com,com]
    A : Normalized technical requirements [com,com]
    G_con : Constructed emissions [ext,com]
    F : Normalized, constructed emissions [ext, com]

    """

    # Default output
    G_con = np.empty(0)
    F = np.empty(0)
    # Basic Variables

    if not E_bar.size:
        E_bar = np.eye(len(V))

    (V_tild, V_bar, _, _) = _rank_products(E_bar, V)

    # The construct
    Z = (U - V_tild).dot(E_bar.T)  # <-- eq:btc
    (A, _, _) = matrix_norm(Z, V_bar)

    if G.size:
        G_con = G.dot(E_bar.T)  # <-- eq:NonProdBalEnvExt
        (F, _, _) = matrix_norm(G_con, V_bar)
    return(Z, A, G_con, F)

###############################################################################
#  Helper functions


def basic_variables(U, V, G=np.empty(0)):
    """ From Use, Supply and Emissions, calculate intermediate variables.

    Parameters
    ----------
    U : Use table [com, ind]
    V : Supply table [com, ind]
    G : Unallocated emissions [ext, ind] (default=np.empty(0))

    Returns
    --------
    com : number of commodities (products)
    ind : number of industries (activities)
    org : number of origin industries (for traceable flows)
    traceable : boolean, are use flows traceable, true or false?
    e_com : vertical vector of ones [com, 1]
    e_ind : vertical vector of ones [ind, 1]
    q : total production volume of each commodity
    g : total production volume of each industry
    ext : number of environmental stressors/factors of production
    """
    # Default values
    ext = np.empty(0)

    # Get basic dimensions
    #
    com = V.shape[0]
    ind = V.shape[1]

    # Extra dimensions and traceability
    #
    if U.ndim == V.ndim:  # untraceable
        traceable = False
        org = 1
    elif U.ndim == V.ndim + 1:  # traceable
        traceable = True
        org = np.size(U, 0)
    else:
        print("Strange dimensions of U and V")

    if G.size:
        ext = np.size(G, 0)

    # Summation vectors
    e_com = np.ones(com)
    e_ind = np.ones(ind)

    # totals
    #
    ## Industry total
    g = np.dot(V.T, e_com)
    ## Product total
    q = np.dot(V, e_ind)

    return (com, ind, org, traceable, e_com, e_ind, q, g, ext)


def _rank_products(E_bar, V=np.empty(0), U=np.empty(0)):
    """Distinguish between primary and secondary products in flow variables

    Parameters
    ----------
    E_bar : 0 or 1 mapping of primary commodities to industries [com,ind]
    U : Use table [com, ind]
    V : Supply table [com, ind]

    Returns
    --------
    V_tild : Table of secondary production flows [com, ind]
    V_bar : Table of primary production flows [com, ind]
    U_tild : Table of Use flows traceable to secondary production
        [ind,com,ind]
    E_tild : 0 or 1 mapping of secondary products to industries [com,ind]

    """

    # Initialize variables
    V_bar = np.empty(0)
    V_tild = np.empty(0)
    U_tild = np.empty(0)

    # E_tild, opposite of E_bar
    E_tild = np.ones(E_bar.shape, dtype=np.int) - E_bar

    # Filtering outputs of V
    if V.size:
        V_bar = V * E_bar
        V_tild = V * E_tild

    # Filtering traceable inputs of U
    if U.ndim == 3:
        ind = U.shape[-1]
        U_tild = np.zeros(U.shape)
        for j in range(ind):
            # Inputs to industry J traceable to secondary production
            U_tild[:, :, j] = U[:, :, j] * E_tild.T

    return(V_tild, V_bar, U_tild, E_tild)

def collapse_dims(x, first2dimensions=False):
    """Collapse 3-d or 4-d array in two dimensions

    Parameters
    ----------
    x : 3d or 4d array to be collapsed

    first2dimensions : Boolean : For 3d array, should the last two dimensions
        be flattened together (default) or should the first two be
        flattened together instead (=true)?

    Returns
    --------
    z : Flatened 2d array

    """

    s = x.shape
    if x.ndim == 4:
        z = x.reshape((s[0] * s[1], s[2] * s[3]))
    elif x.ndim == 3:
        if first2dimensions:
            z = x.reshape((s[0] * s[1], s[2]))
        else:
            z = x.reshape((s[0], s[1] * s[2]))
    elif x.ndim == 2:
        print('Already in 2-dimensional, pass')
        z = x
    else:
        print('PROBLEM? ndim(Y) = {}'.format(x.ndim))
    return z


def matrix_norm(Z, V, keep_fullsize=False):
    """ Normalizes a flow matrix, even if some rows and columns are null

    Parameters
    ----------
    Z : Flow matrix to be normalized
        dimensions : [com, com] | [com, ind,com] | [ind,com,ind,com]
    V : Production volume with which flows are normalized
        [com, ind]

    keep_fullsize: Do not remove empty rows and columns from A, leave with
                   zeros. [Default, false, don't do it]

    Returns
    --------
    A : Normalized flow matrix, without null rows and columns
    nn_in : filter applied to rows (0 for removed rows, 1 for kept rows)
    nn_out : filter applied to cols (0 for removed cols, 1 for kept cols)

    """
    # Collapse dimensions
    if Z.ndim > 2:
        Z = collapse_dims(Z)
    # Basic Variables
    com = np.size(V, 0)
    ind = np.size(V, 1)
    com2 = np.size(Z, 0)

    # Total production, both aggregate and traceable
    q = np.sum(V, 1)
    u = np.sum(Z, 1)
    q_tr = np.zeros(ind * com)
    for i in range(ind):
        q_tr[i * com:(i + 1) * com] = V[:, i]

    # Filter inputs. Preserve only commodities that are used (to get the recipe
    # right) or that are produced (to get the whole matrix square)
    if np.size(Z, 0) == com:
        nn_in = (abs(q) + abs(u)) != 0
    elif np.size(Z, 0) == com * ind:
        nn_in = (abs(q_tr) + abs(u)) != 0
    else:
        nn_in = np.ones(com2, dtype=bool)

    if np.size(Z, 1) == com:
        nn_out = q != 0
        A = Z[nn_in, :][:, nn_out].dot(np.linalg.inv(ddiag(q[nn_out])))
    elif np.size(Z, 1) == (com * ind):
        nn_out = q_tr != 0
        A = Z[nn_in, :][:, nn_out].dot(np.linalg.inv(ddiag(q_tr[nn_out])))
    else:
        nn_out = np.ones(np.size(Z, 1), dtype=bool)
        A = Z.dot(np.linalg.inv(ddiag(q_tr)))

    if keep_fullsize:
        A0 = np.zeros([Z.shape[0], A.shape[1]])
        A1 = np.zeros_like(Z)
        A0[nn_in, :] = A
        A1[:, nn_out] = A0
        A = A1

    # Return
    return (A, nn_in, nn_out)

def diaginv(x):
    """Diagonalizes a vector and inverses it, even if it contains zero values.

    * Element-wise divide a vector of ones by x
    * Replace any instance of Infinity by 0
    * Diagonalize the resulting vector

    Parameters
    ----------
    x : vector to be diagonalized

    Returns
    --------
    y : diagonalized and inversed vector
       Values on diagonal = 1/coefficient, or 0 if coefficient == 0

    """
    y = np.ones(len(x)) / x
    y[y == np.Inf] = 0
    return ddiag(y)

def ddiag(a, nozero=False):
    """ Robust diagonalization : always put selected diagonal on a diagonal!

    This small function aims at getting a behaviour closer to the
    mathematical "hat", compared to what np.diag() can delivers.

    If applied to a vector or a 2d-matrix with one dimension of size 1, put
    the coefficients on the diagonal of a matrix with off-diagonal elements
    equal to zero.

    If applied to a 2d-matrix (with all dimensions of size > 1), replace
    all off-diagonal elements by zeros.

    Parameters
    ----------
    a : numpy matrix or vector to be diagonalized

    Returns
    --------
    b : Diagonalized vector

    Raises:
       ValueError if a is more than 2dimensional

    See Also
    --------
        diag
    """

    # If numpy vector
    if a.ndim == 1:
        b = np.diag(a)

    # If numpy 2d-array
    elif a.ndim == 2:

        #...but with dimension of magnitude 1
        if min(a.shape) == 1:
            b = np.diag(np.squeeze(a))

        # ... or a "true" 2-d matrix
        else:
            b = np.diag(np.diag(a))

    else:
        raise ValueError("Input must be 1- or 2-d")

    # Extreme case: a 1 element matrix/vector
    if b.ndim == 1 & b.size == 1:
        b = b.reshape((1, 1))

    if nozero:
        # Replace offdiagonal zeros by nan if desired
        c = np.empty_like(b) *  np.nan
        di = np.diag_indices_from(c)
        c[di] = b.diagonal()
        return c
    else:
        # A certainly diagonal vector is returned
        return b
