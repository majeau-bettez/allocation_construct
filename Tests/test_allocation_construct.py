"""
test_allocation_construct
===========================

 Performs unit testing for all functions of allocation_construct.py

 Follows closely the examples in the supporting information of:

     Majeau-Bettez, G., R. Wood, and A.H. Strømman. 2014. Unified Theory of
     Allocations and Constructs in Life Cycle Assessment and Input-Output
     Analysis. Journal of Industrial Ecology 18(5): 747–770.
     DOI:10.1111/jiec.12142

 From directory of allocation_construct.py, run:
     python -m unittest -v

 :copyright: 2013, Guillaume Majeau-Bettez, Richard Wood, Anders H. Strømman
 :license: BSD 2-Clause License

"""

import unittest
import allocation_construct as ac
import numpy as np
import numpy.testing as npt
# pylint: disable-msg=C0103

class TestAllocationsConstructs(unittest.TestCase):
    """ Unit test class for allocations and constructs"""

    ## DEFINE SIMPLE TEST CASES
    def setUp(self):
        """
        We define simple Supply and Use Table systems to test all allocations
        anc constructs.

        We first define a Supply and traceable Use Table system, with three
        products (i, j, k) and four industries (I, J1, J2, and K).

        Then, by aggregation we generate a Supply and untraceable Use table
        system (SuUT).

        To test commodity-technology construct (CTC) and the
        byproduct-technology construct (BTC), we also generate a square SUT
        (Va, Ua, Ga), with three products and three industries (I, J, K), by
        further aggregation.


        """

        # absolute tolerance for assertion tests (rouding error)
        self.atol = 1e-08

        # CASE 0
        #---------

        # Defining labels for industries, commodities and factors of prod.
        self.l_ind = np.array([['I', 'J1', 'J2', 'K']], dtype=object).T
        self.l_com = np.array([['i', 'j', 'k']], dtype=object).T
        self.l_ext = np.array([['CO2', 'CH4']], dtype=object).T

        # dimensions
        self.ind = len(self.l_ind)
        self.com = len(self.l_com)

        # labels for traceable flows
        self.l_tr = list()
        for i in self.l_ind:
            for j in self.l_com:
                self.l_tr.append(j + '_{' + i + '}')

        # Supply table
        self.V = np.array([[2, 0, 0, 0],
                           [1, 1, 3, 0],
                           [0, 0, 0, 11]], dtype=float)

        # Traceable use table
        self.Ut = np.array(np.zeros((4, 3, 4), float))
        self.Ut[3, 2, 0] = 4    # use of k from K by I
        self.Ut[3, 2, 1] = 0.75    # use of k from K by J1
        self.Ut[3, 2, 2] = 2    # use of k from K by J2
        self.Ut[0, 1, 3] = 0.25    # use of j from I by K
        self.Ut[2, 1, 3] = 0.5     # use of j from J2 by K

        # Untraceable use table
        self.Uu = np.array(sum(self.Ut, 0))


        # Use of factors of production by industries
        self.G = np.array([
            [10,    4,    15,    18],
            [0,    0,    1,    0]
            ], dtype=float)


        # Intensive properties used in partitioning
        self.PSI = np.array([
        #       I       J1      J2      K
            [0.1,     0.1,     0.1,     0.1],     # i
            [0.2,     0.2,     0.2,     0.2],     #j
            [0.3,     0.3,     0.3,     0.3],     # k
            ])

        # Alternate activity, used in AAA/AAC
        self.Gamma = np.array([
        #    i        j        k
            [1,       0,       0],       # I
            [0,       0,       0],       # J1
            [0,       1,       0],       # J2
            [0,       0,       1]        # K
            ])

        # Identifies primary product of each industry
        self.E_bar = np.array([
        #    I      J1      J2      K
            [1,       0,       0,       0],     #  i
            [0,       1,       1,       0],     #  j
            [0,       0,       0,       1],     #  k
            ])

        # Substitutability between products
        self.Xi = np.array([
            [1,	0,	0],
            [0,	0,	0],
            [0,	0.3,	1]
            ])

        # Square SUT
        self.Va = self.V.dot(self.E_bar.T)
        self.Ua = self.Uu.dot(self.E_bar.T)
        self.Ga = self.G.dot(self.E_bar.T)


    # ---------------------
    # TEST HELPER FUNCTIONS
    # ---------------------
    def test_dimensions_detection_untrace(self):
        """ Tests detection of dimensions and traceability in SuUT"""

        variables = ac.basic_variables(self.Uu, self.V, self.G)
        com = variables[0]
        ind = variables[1]
        org = variables[2]
        ext = variables[-1]
        traceable = variables[3]

        self.assertEqual(com, 3)
        self.assertEqual(ind, 4)
        self.assertEqual(org, 1)
        self.assertEqual(ext, 2)
        self.assertFalse(traceable)

    def test_dimensions_detection_trace(self):
        """ Tests detection of dimensions and traceability in StUT"""

        variables = ac.basic_variables(self.Ut, self.V, self.G)

        com = variables[0]
        ind = variables[1]
        org = variables[2]
        ext = variables[-1]

        traceable = variables[3]

        self.assertEqual(com, 3)
        self.assertEqual(ind, 4)
        self.assertEqual(org, 4)
        self.assertEqual(ext, 2)
        self.assertTrue(traceable)

    def test_summation_vectors(self):
        """ Tests generation of summation vectors of right dimensions"""

        variables = ac.basic_variables(self.Ut, self.V, self.G)
        e_com = variables[4]
        e_ind = variables[5]

        npt.assert_equal(e_com, np.array([1., 1., 1.]))
        npt.assert_equal(e_ind, np.array([1., 1., 1., 1.]))

    def test_totals(self):
        """ Test calculation of total product and industry outputs"""

        variables = ac.basic_variables(self.Ut, self.V, self.G)
        q = variables[6]
        g = variables[7]

        npt.assert_equal(q, np.array([2., 5., 11.]))
        npt.assert_equal(g, np.array([3., 1., 3., 11.]))

    def test_E_tild(self):
        """ Tests generation of filter for secondary production flows"""

        E_tild0 = np.array([[0, 1, 1, 1],
                            [1, 0, 0, 1],
                            [1, 1, 1, 0]])

        __, __, __, E_tild = ac._rank_products(self.E_bar)

        npt.assert_equal(E_tild0, E_tild)


    def test_rank_products_untrace(self):
        """ Tests splitting of primary/secondary flows in a SuUT"""

        V_tild0 = np.zeros((3, 4))
        V_tild0[1, 0] = 1.0
        V_bar0 = np.array([[2.,   0.,   0.,   0.],
                           [0.,   1.,   3.,   0.],
                           [0.,   0.,   0.,  11.]])

        V_tild, V_bar, U_tild, __ = ac._rank_products(self.E_bar,
                                                          self.V,
                                                          self.Uu)

        npt.assert_equal(V_tild, V_tild0)
        npt.assert_equal(V_bar, V_bar0)
        npt.assert_equal(U_tild, np.empty(0))


    def test_rank_products_trace(self):
        """ Tests splitting of primary/secondary use flows in a StUT"""

        U_tild0 = np.zeros((4, 3, 4))
        U_tild0[0, 1, 3] = 0.25

        __, __, U_tild, __ = ac._rank_products(self.E_bar, U=self.Ut)

        npt.assert_equal(U_tild, U_tild0)

    def test_matrix_norm_2d(self):
        """ Test normalization/cleanup of 2D (aggregated flow matrix

        There is 1 null production (second product not produced)."""

        Z = np.array([[0.,  0., 1.],
                      [0.,  0., 0],
                      [3.,  0., 0.5]])

        V = np.array([[4, 1, 0],
                      [0, 0, 0],
                      [0, 0, 2.]])

        A0 = np.array([[0.  ,  0.5 ],
                       [0.6 ,  0.25]])

        A, nn_in, nn_out = ac.matrix_norm(Z, V)

        npt.assert_allclose(A, A0)
        npt.assert_equal(nn_in, nn_out, np.array([True, False, True]))

    def test_matrix_norm_2d_full(self):
        """ Test normalization/cleanup of 2D (aggregated flow matrix

        There is 1 null production (second product not produced)."""

        Z = np.array([[0.,  0., 1.],
                      [0.,  0., 0],
                      [3.,  0., 0.5]])

        V = np.array([[4, 1, 0],
                      [0, 0, 0],
                      [0, 0, 2.]])

        A0 = np.array([[0.,  0, 0.5 ],
                       [0.,  0, 0.0 ],
                       [0.6, 0, 0.25]])

        A, nn_in, nn_out = ac.matrix_norm(Z, V, True)

        npt.assert_allclose(A, A0)
        npt.assert_equal(nn_in, nn_out, np.array([True, False, True]))


    def test_matrix_norm_3d(self):
        """ Test normalization/cleanup of 3D (untraceable) flow matrix"""

        Z = np.array([[[0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.  ]],

                      [[0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.75]],

                      [[3.7 ,  0.  ,  0.  ],
                       [0.  ,  0.75,  0.  ],
                       [0.  ,  2.  ,  0.  ],
                       [0.  ,  0.  ,  0.  ]]])

        A0 = np.array([[0.   , 0.   ,  0.        ,  0.        ,  0.        ],
                       [0.   , 0.   ,  0.        ,  0.        ,  0.06818182],
                       [1.85 , 0.   ,  0.75      ,  0.66666667,  0.        ]])

        nn_out0 = np.array([True, True, False, False, True, False, False,
                            True, False, False, False, True])

        A, nn_in, nn_out = ac.matrix_norm(Z, self.V)

        npt.assert_allclose(A, A0)
        npt.assert_equal(nn_in, np.array([True, True, True]))
        npt.assert_equal(nn_out, nn_out0)

    def test_matrix_norm_4d(self):
        """ Test normalization/cleanup of 4D (traceable) flow matrix"""

        Z0 = np.array([[[[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.75]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[3.7 ,  0.  ,  0.  ],
                         [0.  ,  0.75,  0.  ],
                         [0.  ,  2.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]]]])

        A0 = np.array([
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.06818182],
            [1.85      ,  0.        ,  0.75      ,  0.66666667,  0.        ]])

        A, nn_in, nn_out = ac.matrix_norm(Z0, self.V)
        npt.assert_allclose(A, A0)

    # ----------------------------------------
    # TEST ALLOCATION AND CONSTRUCT FUNCTIONS
    # ----------------------------------------
    def test_partition_coefficients(self):
        """ Test calculation of PA coeff. (PHI) from intensive properties (PSI)
        """

        PHI0 = np.array([[0.5,  0.5,  0. ],
                         [0. ,  1. ,  0. ],
                         [0. ,  1. ,  0. ],
                         [0. ,  0. ,  1. ]])
        PHI = ac.pa_coeff(self.V, self.PSI)
        npt.assert_allclose(PHI0, PHI)

    def test_pa_untrace_psi(self):
        """ Tests partition allocation on SuUT"""

        Z0 = np.array([[[0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ]],

                       [[0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.75]],

                       [[2.  ,  2.  ,  0.  ],
                        [0.  ,  0.75,  0.  ],
                        [0.  ,  2.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ]]])

        A0 = np.array([
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.06818182],
            [1.        ,  2.        ,  0.75      ,  0.66666667,  0.        ]])


        G_all0 = np.array([[[5.,   5.,   0.],
                            [0.,   4.,   0.],
                            [0.,  15.,   0.],
                            [0.,   0.,  18.]],

                           [[0.,   0.,   0.],
                            [0.,   0.,   0.],
                            [0.,   1.,   0.],
                            [0.,   0.,   0.]]])

        F0 = np.array([
            [2.5       ,  5.        ,  4.        ,  5.        ,  1.63636364],
            [0.        ,  0.        ,  0.        ,  0.33333333,  0.        ]])

        Z, A, __, __, G_all, F = ac.pa(self.Uu, self.V, self.PSI, G=self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(G_all0, G_all, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)

    def test_pa_trace(self):
        """ Tests partition allocation on StUT"""

        Z, A, __, __, G_all, F = ac.pa(self.Ut, self.V, self.PSI, G=self.G)

        Z0 = np.array([[[[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.25]],

                        [[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.5 ]],

                        [[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ], [0.  ,  0.  ,  0.  ]],

                        [[2.  ,  2.  ,  0.  ], [0.  ,  0.75,  0.  ],
                         [0.  ,  2.  ,  0.  ], [0.  ,  0.  ,  0.  ]]]])

        A0 = np.array([
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.02272727],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.04545455],
            [1.        ,  2.        ,  0.75      ,  0.66666667,  0.        ]])

        F0 = np.array([
            [2.5       ,  5.        ,  4.        ,  5.        ,  1.63636364],
            [0.        ,  0.        ,  0.        ,  0.33333333,  0.        ]])


        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)


    def test_pc_agg(self):
        """ Tests partition aggregation construct on SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.75],
                       [2.  ,  4.75,  0.  ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.        ,  0.95      ,  0.        ]])

        G_con0 = np.array([[5.,  24.,  18.],
                           [0.,   1.,   0.]])

        F0 = np.array([[2.5       ,  4.8       ,  1.63636364],
                       [0.        ,  0.2       ,  0.        ]])

        Z, A, __, __, G_con, F = ac.pc_agg(self.Uu, self.V, self.PSI, G=self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(G_con0, G_con, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)


    def test_psa_untrace(self):
        """ Tests Product Substition Allocation on SuUT"""

        Z0 = np.array([[[0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ]],

                       [[0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.75]],

                       [[3.7 ,  0.  ,  0.  ],
                        [0.  ,  0.75,  0.  ],
                        [0.  ,  2.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ]]])

        A0 = np.array([[0.        ,  0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.        ,  0.06818182],
                       [1.85      ,  0.75      ,  0.66666667,  0.        ]])

        G_all0 = np.array([[[10.,   0.,   0.],
                            [0.,   4.,   0.],
                            [0.,  15.,   0.],
                            [0.,   0.,  18.]],

                           [[0.,   0.,   0.],
                            [0.,   0.,   0.],
                            [0.,   1.,   0.],
                            [0.,   0.,   0.]]])

        F0 = np.array([[5.        ,  4.        ,  5.        ,  1.63636364],
                       [0.        ,  0.        ,  0.33333333,  0.        ]])

        Z, DeltV, A, __, __, G_all, F = ac.psa(self.Uu, self.V, self.E_bar,
                                               self.Xi, G=self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(G_all0, G_all, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)


    def test_psa_trace(self):
        """ Tests Product Substition Allocation on StUT"""

        Z0 = np.array([[[[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.75]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]]],


                       [[[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]],

                        [[3.7 ,  0.  ,  0.  ],
                         [0.  ,  0.75,  0.  ],
                         [0.  ,  2.  ,  0.  ],
                         [0.  ,  0.  ,  0.  ]]]])


        a = ac.psa(self.Ut, self.V, self.E_bar, self.Xi, self.Gamma, G=self.G)
        Z = a[0]
        npt.assert_allclose(Z0, Z, atol=self.atol)

    def test_psc_agg(self):
        """ Tests Product Substition Construct on SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.75],
                       [3.7 ,  2.75,  0.  ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.85      ,  0.6875    ,  0.        ]])

        F0 = np.array([[5.        ,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        Z, A, __, __, __, F = ac.psc_agg(self.Uu, self.V, self.E_bar,
                                         self.Xi, self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)

    def test_alternate_tech_untrace_simple(self):
        """ Tests construction of Alternate Technology matrix, on simple SuUT
        """

        Ca0 = np.array([[0.        ,  0.        ,  0.        ],
                        [0.        ,  0.        ,  0.06818182],
                        [1.66666667,  0.66666667,  0.        ]])

        Ca, __ = ac.alternate_tech(self.Uu, self.V, self.E_bar, self.Gamma)

        npt.assert_allclose(Ca0, Ca, atol=self.atol)

    def test_alternate_tech_trace_simple(self):
        """ Tests construction of Alternate Technology matrix, on simple StUT
        """

        Ca0 = np.array([[[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.02272727],
                         [0.        ,  0.        ,  0.        ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.04545455],
                         [0.        ,  0.        ,  0.        ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [1.66666667,  0.66666667,  0.        ]]])

        Ca, __  = ac.alternate_tech(self.Ut, self.V, self.E_bar, self.Gamma)
        npt.assert_allclose(Ca0, Ca, atol=self.atol)

    def test_aaa_untrace_simple(self):
        """ Tests Alternate Activity Allocation on SuUT """

        Z0 = np.array([[[0.        ,  0.        ,  0.        ],
                        [0.        ,  0.        ,  0.        ],
                        [0.        ,  0.        ,  0.        ],
                        [0.        ,  0.        ,  0.        ]],

                       [[0.        ,  0.        ,  0.        ],
                        [0.        ,  0.        ,  0.        ],
                        [0.        ,  0.        ,  0.        ],
                        [0.        ,  0.        ,  0.75      ]],

                       [[3.33333333,  0.66666667,  0.        ],
                        [0.        ,  0.75      ,  0.        ],
                        [0.        ,  2.        ,  0.        ],
                        [0.        ,  0.        ,  0.        ]]])

        A0 = np.array([
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.06818182],
            [1.66666667,  0.66666667,  0.75      ,  0.66666667,  0.        ]])

        F0 = np.array([
            [2.5       ,  5.        ,  4.        ,  5.        ,  1.63636364],
            [-0.16666667,  0.33333333,  0.        ,  0.33333333,  0.        ]])

        Z, A, __, __, __, F = ac.aaa(self.Uu, self.V, self.E_bar, self.Gamma,
                                     self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)

    def test_aaa_trace_simple(self):
        """ Tests Alternate Activity Allocation on SuUT """

        Z0 = np.array([[[[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.25      ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]]],


                       [[[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]]],


                       [[[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.5       ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]]],


                       [[[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]],

                        [[0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]],

                        [[3.33333333,  0.66666667,  0.        ],
                         [0.        ,  0.75      ,  0.        ],
                         [0.        ,  2.        ,  0.        ],
                         [0.        ,  0.        ,  0.        ]]]])

        A0 = np.array([
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.02272727],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [0.        ,  0.        ,  0.        ,  0.        ,  0.04545455],
            [1.66666667,  0.66666667,  0.75      ,  0.66666667,  0.        ]])


        Z, A, __, __, __, __ = ac.aaa(self.Ut, self.V, self.E_bar, self.Gamma)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)

    def test_aac_agg(self):
        """ Tests Alternate Activity Construct on SuUT"""

        Z0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.75      ],
                       [3.33333333,  3.41666667,  0.        ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.66666667,  0.68333333,  0.        ]])

        F0 = np.array([[2.5       ,  4.8       ,  1.63636364],
                       [-0.16666667,  0.26666667,  0.        ]])

        Z, A, __, __, __, F = ac.aac_agg(self.Uu, self.V, self.E_bar,
                                         self.Gamma, self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)


    def test_lsa_untrace(self):
        """ Tests Lump Sum Allocation on SuUT"""


        Z0 = np.array([[[0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ]],

                       [[0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ],
                        [0.  ,  0.  ,  0.75]],

                       [[4.  ,  0.  ,  0.  ],
                        [0.  ,  0.75,  0.  ],
                        [0.  ,  2.  ,  0.  ],
                        [0.  ,  0.  ,  0.  ]]])

        A0 = np.array([[0.        ,  0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.        ,  0.06818182],
                       [1.33333333,  0.75      ,  0.66666667,  0.        ]])


        Z, __, A, __, __, __, __ = ac.lsa(self.Uu, self.V, self.E_bar)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)

    def test_lsc(self):
        """ Tests Lump Sum Construct on SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.75],
                       [4.  ,  2.75,  0.  ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.33333333,  0.6875    ,  0.        ]])

        G_con0 = np.array([[10.,  19.,  18.],
                           [0.,   1.,   0.]])

        F0 = np.array([[3.33333333,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        Z, A, __,__, G_con, F = ac.lsc(self.Uu, self.V, self.E_bar, self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(G_con0, G_con, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)

    def test_itc(self):
        """ Tests Industry Technology Construct on SuUT"""

        Z0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.75      ],
                       [2.66666667,  4.08333333,  0.        ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.33333333,  0.81666667,  0.        ]])


        G_con0 = np.array([[6.66666667,  22.33333333,  18.        ],
                           [0.        ,   1.        ,   0.        ]])



        F0 = np.array([[3.33333333,  4.46666667,  1.63636364],
                       [0.        ,  0.2       ,  0.        ]])


        Z, A, G_con, F = ac.itc(self.Uu, self.V, self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(G_con0, G_con, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)

    def test_btc_nonsquare(self):
        """ Tests Byproduct Technology Construct on non-square SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [-1.  ,  0.  ,  0.75],
                       [4.  ,  2.75,  0.  ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [-0.5       ,  0.        ,  0.06818182],
                       [2.        ,  0.6875    ,  0.        ]])

        G_con0 = np.array([[10.,  19.,  18.],
                           [0.,   1.,   0.]])

        F0 = np.array([[5.        ,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        Z, A, G_con, F = ac.btc(self.Uu, self.V, self.E_bar, self.G)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(G_con0, G_con, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)

    def test_btc_square(self):
        """Tests Byproduct Technology Construct on square SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [-1.  ,  0.  ,  0.75],
                       [4.  ,  2.75,  0.  ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [-0.5       ,  0.        ,  0.06818182],
                       [2.        ,  0.6875    ,  0.        ]])

        G_con0 = np.array([[10.,  19.,  18.],
                           [0.,   1.,   0.]])

        F0 = np.array([[5.        ,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        Z, A, G_con, F = ac.btc(self.Ua, self.Va, G=self.Ga)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(G_con0, G_con, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)


    def test_ctc(self):
        """ Tests Commodity Technology Construct on square SuUT"""

        Z0 = np.array([[0.    ,  0.    ,  0.    ],
                       [0.    ,  0.    ,  0.75  ],
                       [3.3125,  3.4375,  0.    ]])


        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.65625   ,  0.6875    ,  0.        ]])


        G_con0 = np.array([[5.25,  23.75,  18.  ],
                           [-0.25,   1.25,   0.  ]])


        F0 = np.array([[2.625     ,  4.75      ,  1.63636364],
                       [-0.125     ,  0.25      ,  0.        ]])

        Z, A, G_con, F = ac.ctc(self.Ua, self.Va, self.Ga)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(G_con0, G_con, atol=self.atol)
        npt.assert_allclose(F0, F, atol=self.atol)

if __name__ == '__main__':
    unittest.main()
