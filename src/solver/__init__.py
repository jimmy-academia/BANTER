from .greedy import RandomSolver, GreedySolver
from .auction import AuctionSolver
from .group import GroupSolver

from .ncf import NCFSolver
from .lightgcn import LightGCNSolver
from .hetrecsys import HetRecSysSolver

from .market import BANTERSolver


def get_solver(args, _method):
    if _method == 'Random':
        return RandomSolver(args)
    if _method == 'Greedy':
        return GreedySolver(args)
    if _method == 'Auction':
        return AuctionSolver(args)
    if _method == 'Group':
        return GroupSolver(args)
    
    if _method == 'NCF':
        return NCFSolver(args)
    if _method == 'LightGCN':
        return LightGCNSolver(args)
    if _method == 'HetRecSys':
        return HetRecSysSolver(args)
    
    if _method == 'BANTER':
        return BANTERSolver(args)
