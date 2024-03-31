from solver import BANTERSolver
from utils import *
from printers.central_plotter import line_plot

def do_case_study():
    case_colors = ['#D62728', '#1770af']
    figsize = (10,7)
    # prepare comparison files
    args = default_args()
    args.setN = None
    args.setM = None
    args.checkpoint_dir = args.ckpt_dir / 'case'
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print()
    print('>>> (case.py) Case study')

    args.nft_project_name = nft_project_name = 'fatapeclub'
    args.breeding_type = 'Heterogeneous'
    Solver = BANTERSolver(args)

    output_subdir = output_dir / 'case'
    output_subdir.mkdir(parents=True, exist_ok=True)

    lineplot1 = output_subdir/'pri_homo_v_child.jpg'
    # directly read files to compare pricing 
    homo_price = torch.load(f'ckpt/main_exp/{nft_project_name}_BANTER_Homogeneous.pth')['pricing']
    child_price = torch.load(f'ckpt/main_exp/{nft_project_name}_BANTER_ChildProject.pth')['pricing']

    numk = len(homo_price)//10 + 1
    sorted_indices = torch.argsort(Solver.Vj, descending=True).cpu()

    if not lineplot1.exists():

        print('Homogeneous vs ChildProject')
        homo_line = []
        child_line = []
        X = []
        for x, batch_ids in enumerate(make_batch_indexes(sorted_indices, numk)):
            homo_line.append(torch.mean(homo_price[batch_ids]).item())
            child_line.append(torch.mean(child_price[batch_ids]).item())
            X.append(torch.mean(Solver.Vj[batch_ids]).item())
        # homo_line.reverse()
        # child_line.reverse()
        # X.reverse()
        infos = {
            'figsize': figsize,
            'ylabel': 'Pricing',
            'xlabel': 'Rarity (V)',
            'colors': case_colors,
            'markers': ['P', 'X'],
            'legends': ['Homogeneous', 'ChildProject'],
        }
        line_plot(X, [homo_line, child_line], infos, lineplot1)

    # find instance example:
    diff = child_price[sorted_indices[:numk]] - homo_price[sorted_indices[:numk]]
    idx = diff.argmax()
    print(sorted_indices[idx], Solver.Vj[sorted_indices[idx]], homo_price[sorted_indices[idx]], child_price[sorted_indices[idx]])

    diff = child_price[sorted_indices[-numk:]] - homo_price[sorted_indices[-numk:]]
    idx = diff.argmin()
    print(sorted_indices[idx-numk], Solver.Vj[sorted_indices[idx-numk]], homo_price[sorted_indices[idx-numk]], child_price[sorted_indices[idx-numk]])


    cachefile = output_subdir/'heter.pth'
    lineplot2 = output_subdir/'pri_niche_v_eclectic.jpg'
    orig_vj = Solver.Vj[Solver.nft_attribute_classes == 0]

    if not lineplot2.exists():
        print('Heterogeneous niche vs eclectic')
        ## percentage of selecting first attribute class selection vs boosting of first attribute class value

        niche_line = []
        eclectic_line = []
        X = [5*x for x in range(11)]
        multiplications = [1+x *0.01 for x in X]

        topk = 10

        if not cachefile.exists():
            for multiple in multiplications:
                Solver.Vj[Solver.nft_attribute_classes == 0] = orig_vj * multiple
                Solver.buyer_types = torch.zeros_like(Solver.buyer_types)
                ranked_parent_nfts, ranked_parent_expectations = Solver.prepare_parent_nfts()

                nft_id_list = ranked_parent_nfts[:, :topk].flatten()
                nft_class_list = Solver.nft_attribute_classes[nft_id_list]
                niche_line.append((sum(nft_class_list == 0)/ len(nft_class_list)).item())

                Solver.buyer_types = torch.ones_like(Solver.buyer_types)
                ranked_parent_nfts, ranked_parent_expectations = Solver.prepare_parent_nfts()
                nft_id_list = ranked_parent_nfts[:, :topk].flatten()
                nft_class_list = Solver.nft_attribute_classes[nft_id_list]
                eclectic_line.append((sum(nft_class_list == 0)/ len(nft_class_list)).item())
            torch.save([niche_line, eclectic_line], cachefile)
        else:
            niche_line, eclectic_line = torch.load(cachefile)

        niche_line, eclectic_line = map(lambda _list: [100*x for x in _list], [niche_line, eclectic_line])        

        infos = {
            'figsize': figsize,
            'ylabel': 'Percentage (%)',
            'xlabel': 'Value boost (%)',
            'colors': case_colors,
            'markers': ['P', 'X'],
            'legends': ['all niche', 'all eclectic'],
        }
        line_plot(X, [niche_line, eclectic_line], infos, lineplot2)


    # find all first class buyer id
    Solver.Vj[Solver.nft_attribute_classes == 0] = orig_vj
    Solver.buyer_types = torch.zeros_like(Solver.buyer_types)
    aranked_parent_nfts, aranked_parent_expectations = Solver.prepare_parent_nfts()
    Solver.buyer_types = torch.ones_like(Solver.buyer_types)
    cranked_parent_nfts, cranked_parent_expectations = Solver.prepare_parent_nfts()


    Solver.Vj[Solver.nft_attribute_classes == 0] = orig_vj * 1.5

    Solver.buyer_types = torch.zeros_like(Solver.buyer_types)
    branked_parent_nfts, branked_parent_expectations = Solver.prepare_parent_nfts()
    Solver.buyer_types = torch.ones_like(Solver.buyer_types)
    dranked_parent_nfts, dranked_parent_expectations = Solver.prepare_parent_nfts()

    print(f'''
    {Solver.nft_attribute_classes[aranked_parent_nfts[829,0:2]]}
    {Solver.nft_attribute_classes[branked_parent_nfts[829,0:2]]}
    {aranked_parent_nfts[829,0:2]}
    {branked_parent_nfts[829,0:2]}

    {Solver.nft_attribute_classes[cranked_parent_nfts[829,0:2]]}
    {Solver.nft_attribute_classes[dranked_parent_nfts[829,0:2]]}
    {cranked_parent_nfts[829,0:2]}
    {dranked_parent_nfts[829,0:2]}
    ''')

