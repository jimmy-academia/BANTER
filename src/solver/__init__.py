from .heuristics import RandomSolver, PopularSolver
from .optimization import AuctionSolver
from .group import GroupSolver
from .recsys import HetRecSysSolver
from .market import BANTERSolver

def get_solver(args, _method):
    if _method == 'Random':
        return RandomSolver(args)
    if _method == 'Popular':
        return PopularSolver(args)
    if _method == 'Auction':
        return AuctionSolver(args)
    if _method == 'Group':
        return GroupSolver(args)
    if _method == 'HetRecSys':
        return HetRecSysSolver(args)
    if _method == 'BANTER':
        return BANTERSolver(args)
