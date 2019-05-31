from dataloaders.datasets import cityscapes, combine_dbs, pascal, sbd, lfw, celebA
from torch.utils.data import DataLoader, ConcatDataset

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'lfw':
        train_set = lfw.LFWSegmentation(args, split='train')
        val_set = lfw.LFWSegmentation(args, split='val')
        test_set = lfw.LFWSegmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'celebA':
        train_set = celebA.CelebASegmentation(args, split='train')
        val_set = celebA.CelebASegmentation(args, split='val')
        test_set = celebA.CelebASegmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'lfw_celebA':
        train_set_celebA = celebA.CelebASegmentation(args, split='train')
        val_set_celebA = celebA.CelebASegmentation(args, split='val')
        test_set_celebA = celebA.CelebASegmentation(args, split='test')

        train_set_lfw = lfw.LFWSegmentation(args, split='train')
        val_set_lfw = lfw.LFWSegmentation(args, split='val')
        test_set_lfw = lfw.LFWSegmentation(args, split='test')

        num_class = train_set_celebA.NUM_CLASSES
        train_set = ConcatDataset([train_set_celebA, train_set_lfw])
        val_set = ConcatDataset([val_set_celebA, val_set_lfw])
        test_set = ConcatDataset([test_set_celebA, test_set_lfw])

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class 
    
    else:
        raise NotImplementedError

