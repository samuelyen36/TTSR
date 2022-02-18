from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):
    ### import module
    m = import_module('dataset.' + args.dataset.lower())        #import something from dataset.cufed.py

    if (args.dataset == 'CUFED'):
        data_train = getattr(m, 'TrainSet')(args)
        data_multi = getattr(m, "TestSet_multiframe")(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):  #five reference frame, use different dataloader?
            data_test = getattr(m, 'TestSet')(args=args, ref_level=str(i+1))
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)        #1~5
        dataloader_test_multi = DataLoader(data_multi, batch_size=1, shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test, 'multi': dataloader_test_multi}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader