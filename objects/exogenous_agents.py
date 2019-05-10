
class CentralBank:
    """
    Class holding central bank properties
    """

    def __init__(self, variables, parameters):
        self.var = variables
        self.par = parameters

    def __repr__(self):
        return 'central_bank_' + str(self.par.country)


class CBVariables:
    """Holds the inital variables for the central bank"""
    def __init__(self, assets, currency, asset_demand, asset_target):
        self.assets = [assets] # The quantities of assets held
        self.currency = [currency] # The quantities of currencies held
        self.asset_demand = [asset_demand]
        self.asset_target = asset_target


class CBParameters:
    """Holds the inital variables for the central bank"""
    def __init__(self, reserve_rate):
        self.reserve_rate = reserve_rate

