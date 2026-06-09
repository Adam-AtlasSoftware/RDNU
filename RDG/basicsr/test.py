import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        if hasattr(test_loader, 'dataset'):
            test_set_name = test_loader.dataset.opt['name']
        else:
            test_set_name = 'Unknown'
        logger.info(f'Testing {test_set_name}...')
        
        # If it's a DALIWrapper, we can just iterate over it
        if test_loader.__class__.__name__ == 'DALIWrapper':
            import cv2
            import os
            from basicsr.utils import tensor2img
            import numpy as np
            
            save_dir = osp.join(opt['path']['visualization'], test_set_name)
            os.makedirs(save_dir, exist_ok=True)
            
            model.net_g.eval()
            for batch_idx, val_data in enumerate(test_loader):
                if batch_idx >= 5: # Just test a few batches to see the results
                    break
                    
                model.feed_data(val_data)
                
                with torch.no_grad():
                    model.output = model.net_g(model.lq)
                    
                    # If this is test phase, output will be 5120x2880 (from 720p 4x upscale)
                    # We want to downscale this to 1080p (1920x1080) for comparison
                    import torch.nn.functional as F
                    b, t, c, h, w = model.output.shape
                    if h == 2880 and w == 5120:
                        out_flat = model.output.view(-1, c, h, w)
                        out_flat = F.interpolate(out_flat, size=(1080, 1920), mode='bicubic', align_corners=False)
                        out_flat = torch.clamp(out_flat, 0.0, 1.0)
                        model.output = out_flat.view(b, t, c, 1080, 1920)
                
                # Save input LQ and output images
                visuals = model.get_current_visuals()
                
                # `visuals['result']` is [B, T, C, H, W]
                # `visuals['lq']` is [B, T, C, H, W]
                for b_idx in range(visuals['result'].size(0)):
                    for t_idx in range(visuals['result'].size(1)):
                        res_tensor = visuals['result'][b_idx, t_idx, :, :, :]
                        res_img = tensor2img(res_tensor)
                        
                        lq_tensor = visuals['lq'][b_idx, t_idx, :, :, :]
                        lq_img = tensor2img(lq_tensor)
                        
                        cv2.imwrite(osp.join(save_dir, f"batch{batch_idx}_b{b_idx}_t{t_idx}_lq.png"), lq_img)
                        cv2.imwrite(osp.join(save_dir, f"batch{batch_idx}_b{b_idx}_t{t_idx}_out4k.png"), res_img)
                        
                        if 'gt' in visuals and visuals['gt'] is not None:
                            gt_tensor = visuals['gt'][b_idx, t_idx, :, :, :]
                            gt_img = tensor2img(gt_tensor)
                            cv2.imwrite(osp.join(save_dir, f"batch{batch_idx}_b{b_idx}_t{t_idx}_gt.png"), gt_img)
                        
            logger.info("Done testing M3VIR with DALI loader.")
            continue

        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
