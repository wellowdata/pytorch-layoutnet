import os
import glob
import argparse
import numpy as np
import torchfile
from PIL import Image

import torch
from model import LayoutNet
from utils_eval import augment, augment_undo
from pano import get_ini_cor, draw_boundary_from_cor_id
from pano_opt import optimize_cor_id

'''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Model related arguments
parser.add_argument('--path_prefix', default='ckpt/pre',
                    help='prefix path to load model.')
parser.add_argument('--device', default='cuda:0',
                    help='device to run models.')
# I/O related arguments
parser.add_argument('--img_glob', required=True,
                    help='NOTE: Remeber to quote your glob path.')
parser.add_argument('--line_glob', required=True,
                    help='shold have the same number of files as img_glob. '
                         'two list with same index are load as input channels. '
                         'NOTE: Remeber to quote your glob path.')
parser.add_argument('--output_dir', required=True)
# Data augmented arguments (to improve output quality)
parser.add_argument('--flip', action='store_true',
                    help='whether to perfome left-right flip. '
                         '# of input x2.')
parser.add_argument('--rotate', nargs='*', default=[], type=float,
                    help='whether to perfome horizontal rotate. '
                         'each elements indicate fraction of image width. '
                         '# of input xlen(rotate).')
# Post porcessing related arguments
parser.add_argument('--d1', default=21, type=int,
                    help='Post-processing parameter.')
parser.add_argument('--d2', default=3, type=int,
                    help='Post-processing parameter.')
parser.add_argument('--post_optimization', action='store_true',
                    help='whether to performe post gd optimization')
args = parser.parse_args()
device = torch.device(args.device)
'''


def visual_persp(img_glob, output_dir, device=torch.device("cpu"), path_prefix='ckpt/pre',flip=True,  rotate=[], d1=21, d2=3):

    # Check input arguments validation
    for path in glob.glob(img_glob):
        assert os.path.isfile(path), '%s not found' % path
    assert os.path.isdir(output_dir), '%s is not a directory' % output_dir
    for rotate in rotate:
        assert 0 <= rotate and rotate <= 1, 'elements in --rotate should in [0, 1]'


    # Prepare model
    layoutnet = LayoutNet().to(device)
    torch_pretrained = torchfile.load('/home/jupyter/Shapes.ai/pytorch-layoutnet/ckpt/perspfull_lsun_type_pretrained.t7')
    total_parameter =0
    for p in layoutnet.parameters():
        total_parameter += np.prod(p.size())

    torch.save(layoutnet.state_dict(), '/ckpt/persp_pretrained.pth')
        
    # Load path to visualization
    img_paths = sorted(glob.glob(img_glob))


    # Process each input
    for i_path in img_paths:
        print('img  path:', i_path)

        # Load and cat input images
        i_img = np.array(Image.open(i_path).resize((512,512)), np.float32) / 255
        #l_img = np.array(Image.open(l_path), np.float32) / 255
        x_img = i_img.transpose([2, 0, 1])
        print('x_img shape: ', x_img.shape)
        
        # Augment data
        x_imgs_augmented, aug_type = augment(x_img, flip, rotate)

        # Feedforward and extract output images
        with torch.no_grad():
            x = torch.FloatTensor(x_imgs_augmented).to(device)
            print('x shape: ', x.shape)
            deconv6_sf, deconv6_sf_c, ref4 = layoutnet(x) 
            #edg_de_list = edg_decoder(en_list[::-1])
            #cor_de_list = cor_decoder(en_list[-1:] + edg_de_list[:-1])

            edg_tensor = deconv6_sf
            cor_tensor = deconv6_sf_c
            roomtype_tensor = ref4

            # Recover the effect from augmentation
            edg_img = augment_undo(edg_tensor.cpu().numpy(), aug_type)
            cor_img = augment_undo(cor_tensor.cpu().numpy(), aug_type)


        # Merge all results from augmentation
        edgmap = edg_img.transpose([0, 2, 3, 1]).mean(0).copy()
        cormap = cor_img.transpose([0, 2, 3, 1]).mean(0)[..., 0].copy()

        # Post processing to extract layout
        cor_id = get_ini_cor(cormap, d1, d2)
        #if post_optimization:
        #    cor_id = optimize_cor_id(cor_id, edgmap, cormap,
        #                             num_iters=100, verbose=False)

        # Draw extracted layout on source image
        bon_img = draw_boundary_from_cor_id(cor_id.copy(), i_img * 255)

        # Composite all result in one image
        all_in_one = 0.3 * edgmap + 0.3 * cormap[..., None] + 0.4 * i_img
        all_in_one = draw_boundary_from_cor_id(cor_id.copy(), all_in_one * 255)

        # Dump results
        basename = os.path.splitext(os.path.basename(i_path))[0]
        path_edg = os.path.join(output_dir, '%s_edg.png' % basename)
        path_cor = os.path.join(output_dir, '%s_cor.png' % basename)
        path_bon = os.path.join(output_dir, '%s_bon.png' % basename)
        path_all_in_one = os.path.join(output_dir, '%s_all.png' % basename)
        path_cor_id = os.path.join(output_dir, '%s_cor_id.txt' % basename)

        Image.fromarray((edgmap * 255).astype(np.uint8)).save(path_edg)
        Image.fromarray((cormap * 255).astype(np.uint8)).save(path_cor)
        Image.fromarray(bon_img).save(path_bon)
        Image.fromarray(all_in_one).save(path_all_in_one)
        with open(path_cor_id, 'w') as f:
            for x, y in cor_id:
                f.write('%.6f %.6f\n' % (x, y))

            
if __name__ == "__main__":
    img_glob='assert/ADE_val_00000064.jpg'
    output_dir='assert/output/'
    visual_persp(img_glob, output_dir,)