import numpy as np
import sys
import pandas as pd


def portfolio_optimization(trader, day):
    """
    Calculate optimal weights of assets given covariance, return expectations and risk aversion
    :param trader: trader which tries to optimize its portfolio
    :param day: period at which the optimization takes place
    :return:
    """
    # create a copy of the covariance matrix of the funds
    covariance_assets = trader.var.covariance_matrix.copy()

    expected_return_assets = np.zeros((len(covariance_assets)))

    # compute the risk aversion matrix
    risk_aversion_mat = pd.DataFrame((np.zeros((len(covariance_assets),
                                                len(covariance_assets))) + trader.par.risk_aversion)
                                     , index=covariance_assets.columns, columns=covariance_assets.columns)

    # multiply covariance with asset specific risk aversion
    covariance_assets = np.multiply(covariance_assets, risk_aversion_mat)
    original_cov = np.array(covariance_assets)

    # Create a 1D numpy array with one expected returns per asset
    for i, a in enumerate(covariance_assets.columns.values):
        expected_return_assets[i] = trader.exp.returns[a]

    # adding the budget constraint to the covariance matrix - to solve the model using linear algebra
    # Adding a row with ones
    aux_cov = np.concatenate((covariance_assets, np.ones((1, len(covariance_assets)))), axis=0)

    # Adding a column with ones
    aux_cov = np.concatenate((aux_cov, np.ones((len(aux_cov), 1))), axis=1)
    aux_cov[len(aux_cov) - 1, len(aux_cov) - 1] = 0

    # adding the budget constraint to the return vector - to solve the model using linear algebra
    aux_x = np.append(expected_return_assets, 0)
    aux_y = np.zeros(len(aux_x))
    aux_y[len(aux_x) - 1] = 1

    o_aux_x = aux_x.copy()
    o_aux_y = aux_y.copy()
    o_aux_cov = aux_cov.copy()

    kt_count = 0
    test_KT = np.zeros(len(aux_x) - 1)

    while sum(test_KT) != len(test_KT):
        if kt_count > (len(aux_x)-1):
            print("portfolio optimization failed on day ", day)
            # TODO debug
            weights = simple_portfolio_optimization(trader)
            break

        # compute matrix inverse
        try:
            inv_aux_cov = np.linalg.inv(aux_cov)
        except:
            print('error')

        inv_aux_cov = np.linalg.inv(aux_cov)
        aux_c = np.matmul(inv_aux_cov, aux_x)
        aux_d = np.matmul(inv_aux_cov, aux_y)

        # solving for optimal weights
        weights = aux_c + aux_d

        # Start of algorithm that takes out shorted assets
        test = weights[:-1] < -1e-10

        while sum(test) > 0:
            for i in range(len(aux_cov) - 1):
                for j in range(len(aux_cov)):
                    if weights[i] < 0 and i != j:
                        aux_cov[i, j] = 0
                    if weights[i] < 0 and i == j:
                        aux_cov[i, j] = 1
            for i in range(len(aux_x) - 1):
                if weights[i] < 0:
                    aux_x[i] = 0
                    aux_y[i] = 0

            # compute matrix inverse
            inv_aux_cov = np.linalg.inv(aux_cov)
            aux_c = np.matmul(inv_aux_cov, aux_x)
            aux_d = np.matmul(inv_aux_cov, aux_y)

            weights = aux_c + aux_d
            test = weights[:-1] < -1e-10

        aux_e = np.zeros(len(aux_c) - 1)
        aux_f = np.zeros(len(aux_c) - 1)
        for i in range(len(aux_e)):
            aux_e[i] = expected_return_assets[i] - sum([original_cov[i, j] * aux_c[j] for j in range(len(aux_e))]) - aux_c[-1]
            aux_f[i] = sum([original_cov[i, j] * aux_d[j] for j in range(len(aux_e))]) + aux_d[-1]

        aux_kt_pd = aux_e - aux_f  # partial derivatives
        for i in range(len(aux_kt_pd)):
            if weights[i] > 0:
                test_KT[i] = 1
            else:
                test_KT[i] = aux_kt_pd[i] <= sys.float_info.epsilon

        # if Kuhn-Tucker conditions are not fulfilled put the asset with the highest partial derivative (marginal utility) back
        if sum(test_KT) < len(test_KT):
            kt_count = kt_count + 1
            test_KT_inv = 1 - test_KT
            aux2_KT_pd = aux_kt_pd * test_KT_inv
            aux2_KT_pd = aux2_KT_pd.tolist()
            max_pd = max(aux2_KT_pd)
            i = aux2_KT_pd.index(max_pd)
            if test_KT[i] == 0:
                aux_x[i] = o_aux_x[i]
                aux_y[i] = o_aux_y[i]
                for j in range(len(aux_cov)):
                    aux_cov[i, j] = o_aux_cov[i, j]

    output = {}

    for i, a in enumerate(covariance_assets.columns.values):
        output[a] = weights[i]

    return output


def simple_portfolio_optimization(trader):
    # create a copy of the covariance matrix of the funds
    Cov_assets = trader.var.covariance_matrix.copy()
    #
    E_ret_assets = np.zeros((len(Cov_assets)))

    # compute the risk aversion matrix
    risk_aversion_mat = pd.DataFrame((np.zeros((len(Cov_assets),
                                                len(Cov_assets))) + trader.par.risk_aversion)
                                     , index=Cov_assets.columns, columns=Cov_assets.columns)

    # multiply covariance with asset specific risk aversion
    Cov_assets = np.multiply(Cov_assets, risk_aversion_mat)

    # Create a 1D numpy array with one expected returns per asset
    for i, a in enumerate(Cov_assets.columns.values):
        E_ret_assets[i] = trader.exp.returns[a]

    # adding the budget constraint to the covariance matrix - to solve the model using linear algebra
    # Adding a row with ones
    aux_cov = np.concatenate((Cov_assets, np.ones((1, len(Cov_assets)))), axis=0)

    # Adding a column with ones
    aux_cov = np.concatenate((aux_cov, np.ones((len(aux_cov), 1))), axis=1)

    aux_cov[len(aux_cov) - 1, len(aux_cov) - 1] = 0

    # adding the budget constraint to the return vector - to solve the model using linear algebra
    aux_ret = np.append(E_ret_assets, 1)

    # compute matrix inverse
    inv_aux_cov = np.linalg.inv(aux_cov)

    # solving for optimal weights
    weights = np.matmul(inv_aux_cov, aux_ret)
    original_weights = weights.copy()

    # Start of algorithm that takes out shorted assets
    test = weights[:-1] < 0

    ###########################################################################
    ################## COMPUTING ALL SOLUTION PATHS ###########################
    ############################################################################
    tree = {}
    counter = {}
    stored_aux_cov = {}
    stored_aux_ret = {}
    stored_test = {}
    stored_weights = {}
    shorted = []
    layer = 0
    counter[layer] = 0
    solution_number = 0

    for ind, val in enumerate(test):
        if val == True:
            shorted.append(ind)

    tree[layer] = shorted
    stored_test[layer] = test.copy()
    stored_weights[layer] = weights.copy()
    stored_aux_cov[layer] = aux_cov.copy()
    stored_aux_ret[layer] = aux_ret.copy()
    solution_path = {}
    solution_weights = {}

    while layer >= 0:
        while sum(test) > 0:
            aux_cov[tree[layer][counter[layer]], :] = 0
            aux_cov[tree[layer][counter[layer]], tree[layer][counter[layer]]] = 1
            aux_ret[tree[layer][counter[layer]]] = 0
            # recompute compute matrix inverse and weights
            inv_aux_cov = np.linalg.inv(aux_cov)
            weights = np.matmul(inv_aux_cov, aux_ret)
            test = weights[:-1] < -1e-10
            layer += 1

            shorted = []
            for ind, val in enumerate(test):
                if val == True:
                    shorted.append(ind)
            tree[layer] = shorted
            stored_test[layer] = test.copy()
            stored_weights[layer] = weights.copy()
            stored_aux_cov[layer] = aux_cov.copy()
            stored_aux_ret[layer] = aux_ret.copy()
            counter[layer] = 0

        # storing the a possible solution
        solution_path[solution_number] = [tree[l][counter[l]] for l in range(layer)]
        solution_weights[solution_number] = weights
        solution_number += 1

        layer -= 1
        # check if we are on the the highest level: if we are, all solutions have been stored
        if layer < 0:
            break

        aux = False
        while aux == False:
            # check if all solutions of one layer have been computed
            if counter[layer] == len(tree[layer]) - 1 and layer > 0:  # if yes, go back one layer
                layer -= 1
            if counter[layer] != len(tree[
                                         layer]) - 1 and layer >= 0:  # if no, increase the counter of that layer and restore variables of the layer
                counter[layer] += 1
                test = stored_test[layer]
                weights = stored_weights[layer].copy()
                aux_cov = stored_aux_cov[layer].copy()
                aux_ret = stored_aux_ret[layer].copy()
                aux = True
            if layer == 0:
                aux = True

    ##############################################################
    ##############################################################

    # compute utility
    U = {}
    Utility = []
    for sol in range(len(solution_weights)):
        for i in range(len(E_ret_assets)):
            U[i] = solution_weights[sol][i] * E_ret_assets[i] - 0.5 * sum(
                [(solution_weights[sol][j] * solution_weights[sol][i]) * Cov_assets.iloc[i, j] for j in
                 range(len(E_ret_assets))])
        Utility.append((sum(U.values())))

    weights = solution_weights[Utility.index(max(Utility))]

    output = {}

    for i, a in enumerate(Cov_assets.columns.values):
        output[a] = weights[i]

    return weights
