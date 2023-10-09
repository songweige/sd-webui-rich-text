import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision

from scripts.models.utils.richtext_utils import seed_everything
from sklearn.cluster import KMeans, SpectralClustering

# SelfAttentionLayers = [
#     # 'down_blocks.0.attentions.0.transformer_blocks.0.attn1',
#     # 'down_blocks.0.attentions.1.transformer_blocks.0.attn1',
#     'down_blocks.1.attentions.0.transformer_blocks.0.attn1',
#     # 'down_blocks.1.attentions.1.transformer_blocks.0.attn1',
#     'down_blocks.2.attentions.0.transformer_blocks.0.attn1',
#     'down_blocks.2.attentions.1.transformer_blocks.0.attn1',
#     'mid_block.attentions.0.transformer_blocks.0.attn1',
#     'up_blocks.1.attentions.0.transformer_blocks.0.attn1',
#     'up_blocks.1.attentions.1.transformer_blocks.0.attn1',
#     'up_blocks.1.attentions.2.transformer_blocks.0.attn1',
#     # 'up_blocks.2.attentions.0.transformer_blocks.0.attn1',
#     'up_blocks.2.attentions.1.transformer_blocks.0.attn1',
#     # 'up_blocks.2.attentions.2.transformer_blocks.0.attn1',
#     # 'up_blocks.3.attentions.0.transformer_blocks.0.attn1',
#     # 'up_blocks.3.attentions.1.transformer_blocks.0.attn1',
#     # 'up_blocks.3.attentions.2.transformer_blocks.0.attn1',
# ]

SelfAttentionLayers = [
    # 'down_blocks.0.attentions.0.transformer_blocks.0.attn1',
    # 'down_blocks.0.attentions.1.transformer_blocks.0.attn1',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1',
    # 'down_blocks.1.attentions.1.transformer_blocks.0.attn1',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1',
    'mid_block.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1',
    # 'up_blocks.2.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1',
    # 'up_blocks.2.attentions.2.transformer_blocks.0.attn1',
    # 'up_blocks.3.attentions.0.transformer_blocks.0.attn1',
    # 'up_blocks.3.attentions.1.transformer_blocks.0.attn1',
    # 'up_blocks.3.attentions.2.transformer_blocks.0.attn1',
]


CrossAttentionLayers = [
    # 'down_blocks.0.attentions.0.transformer_blocks.0.attn2',
    # 'down_blocks.0.attentions.1.transformer_blocks.0.attn2',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn2',
    # 'down_blocks.1.attentions.1.transformer_blocks.0.attn2',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn2',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn2',
    'mid_block.attentions.0.transformer_blocks.0.attn2',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn2',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn2',
    # 'up_blocks.2.attentions.0.transformer_blocks.0.attn2',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn2',
    # 'up_blocks.2.attentions.2.transformer_blocks.0.attn2',
    # 'up_blocks.3.attentions.0.transformer_blocks.0.attn2',
    # 'up_blocks.3.attentions.1.transformer_blocks.0.attn2',
    # 'up_blocks.3.attentions.2.transformer_blocks.0.attn2'
]

# CrossAttentionLayers = [
#     'down_blocks.0.attentions.0.transformer_blocks.0.attn2',
#     'down_blocks.0.attentions.1.transformer_blocks.0.attn2',
#     'down_blocks.1.attentions.0.transformer_blocks.0.attn2',
#     'down_blocks.1.attentions.1.transformer_blocks.0.attn2',
#     'down_blocks.2.attentions.0.transformer_blocks.0.attn2',
#     'down_blocks.2.attentions.1.transformer_blocks.0.attn2',
#     'mid_block.attentions.0.transformer_blocks.0.attn2',
#     'up_blocks.1.attentions.0.transformer_blocks.0.attn2',
#     'up_blocks.1.attentions.1.transformer_blocks.0.attn2',
#     'up_blocks.1.attentions.2.transformer_blocks.0.attn2',
#     'up_blocks.2.attentions.0.transformer_blocks.0.attn2',
#     'up_blocks.2.attentions.1.transformer_blocks.0.attn2',
#     'up_blocks.2.attentions.2.transformer_blocks.0.attn2',
#     'up_blocks.3.attentions.0.transformer_blocks.0.attn2',
#     'up_blocks.3.attentions.1.transformer_blocks.0.attn2',
#     'up_blocks.3.attentions.2.transformer_blocks.0.attn2'
# ]

# CrossAttentionLayers_XL = [
#     'up_blocks.0.attentions.0.transformer_blocks.1.attn2',
#     'up_blocks.0.attentions.0.transformer_blocks.2.attn2',
#     'up_blocks.0.attentions.0.transformer_blocks.3.attn2',
#     'up_blocks.0.attentions.0.transformer_blocks.4.attn2',
#     'up_blocks.0.attentions.0.transformer_blocks.5.attn2',
#     'up_blocks.0.attentions.0.transformer_blocks.6.attn2',
#     'up_blocks.0.attentions.0.transformer_blocks.7.attn2',
# ]
CrossAttentionLayers_XL = [
    'down_blocks.2.attentions.1.transformer_blocks.3.attn2',
    'down_blocks.2.attentions.1.transformer_blocks.4.attn2',
    'mid_block.attentions.0.transformer_blocks.0.attn2',
    'mid_block.attentions.0.transformer_blocks.1.attn2',
    'mid_block.attentions.0.transformer_blocks.2.attn2',
    'mid_block.attentions.0.transformer_blocks.3.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.1.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.2.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.3.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.4.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.5.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.6.attn2',
    'up_blocks.0.attentions.0.transformer_blocks.7.attn2',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2'
]

def split_attention_maps_over_steps(attention_maps):
    r"""Function for splitting attention maps over steps.
    Args:
        attention_maps (dict): Dictionary of attention maps.
        sampler_order (int): Order of the sampler.
    """
    # This function splits attention maps into unconditional and conditional score and over steps

    attention_maps_cond = dict()    # Maps corresponding to conditional score
    attention_maps_uncond = dict()  # Maps corresponding to unconditional score

    for layer in attention_maps.keys():

        for step_num in range(len(attention_maps[layer])):
            if step_num not in attention_maps_cond:
                attention_maps_cond[step_num] = dict()
                attention_maps_uncond[step_num] = dict()

            attention_maps_uncond[step_num].update(
                {layer: attention_maps[layer][step_num][:1]})
            attention_maps_cond[step_num].update(
                {layer: attention_maps[layer][step_num][1:2]})

    return attention_maps_cond, attention_maps_uncond


def save_attention_heatmaps(attention_maps, tokens_vis, save_dir, prefix):
    r"""Function to plot heatmaps for attention maps.

    Args:
        attention_maps (dict): Dictionary of attention maps per layer
        save_dir (str): Directory to save attention maps
        prefix (str): Filename prefix for html files

    Returns:
        Heatmaps, one per sample.
    """

    html_names = []

    idx = 0
    html_list = []

    for layer in attention_maps.keys():
        if idx == 0:
            # import ipdb;ipdb.set_trace()
            # create a set of html files.

            batch_size = attention_maps[layer].shape[0]

            for sample_num in range(batch_size):
                # html path
                html_rel_path = os.path.join('sample_{}'.format(
                    sample_num), '{}.html'.format(prefix))
                html_names.append(html_rel_path)
                html_path = os.path.join(save_dir, html_rel_path)
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
                html_list.append(open(html_path, 'wt'))
                html_list[sample_num].write(
                    '<html><head></head><body><table>\n')

        for sample_num in range(batch_size):

            save_path = os.path.join(save_dir, 'sample_{}'.format(sample_num),
                                     prefix, 'layer_{}'.format(layer)) + '.jpg'
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

            layer_name = 'layer_{}'.format(layer)
            html_list[sample_num].write(
                f'<tr><td><h1>{layer_name}</h1></td></tr>\n')

            prefix_stem = prefix.split('/')[-1]
            relative_image_path = os.path.join(
                prefix_stem, 'layer_{}'.format(layer)) + '.jpg'
            html_list[sample_num].write(
                f'<tr><td><img src=\"{relative_image_path}\"></td></tr>\n')

            plt.figure()
            plt.clf()
            nrows = 2
            ncols = 7
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

            fig.set_figheight(8)
            fig.set_figwidth(28.5)

            # axs[0].set_aspect('equal')
            # axs[1].set_aspect('equal')
            # axs[2].set_aspect('equal')
            # axs[3].set_aspect('equal')
            # axs[4].set_aspect('equal')
            # axs[5].set_aspect('equal')

            cmap = plt.get_cmap('YlOrRd')

            for rid in range(nrows):
                for cid in range(ncols):
                    tid = rid*ncols + cid
                    # import ipdb;ipdb.set_trace()
                    attention_map_cur = attention_maps[layer][sample_num, :, :, tid].numpy(
                    )
                    vmax = float(attention_map_cur.max())
                    vmin = float(attention_map_cur.min())
                    sns.heatmap(
                        attention_map_cur, annot=False, cbar=False, ax=axs[rid, cid],
                        cmap=cmap, vmin=vmin, vmax=vmax
                    )
                    axs[rid, cid].set_xlabel(tokens_vis[tid])

            # axs[0].set_xlabel('Self attention')
            # axs[1].set_xlabel('Temporal attention')
            # axs[2].set_xlabel('T5 text attention')
            # axs[3].set_xlabel('CLIP text attention')
            # axs[4].set_xlabel('CLIP image attention')
            # axs[5].set_xlabel('Null text token')

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            # fig.colorbar(sm, cax=axs[6])

            fig.tight_layout()
            plt.savefig(save_path, dpi=64)
            plt.close('all')

        if idx == (len(attention_maps.keys()) - 1):
            for sample_num in range(batch_size):
                html_list[sample_num].write('</table></body></html>')
                html_list[sample_num].close()

        idx += 1

    return html_names


def create_recursive_html_link(html_path, save_dir):
    r"""Function for creating recursive html links.
    If the path is dir1/dir2/dir3/*.html,
    we create chained directories
        -dir1
            dir1.html (has links to all children)
            -dir2
                dir2.html   (has links to all children)
                -dir3
                    dir3.html

    Args:
        html_path (str): Path to html file.
        save_dir (str): Save directory.
    """

    html_path_split = os.path.splitext(html_path)[0].split('/')
    if len(html_path_split) == 1:
        return

    # First create the root directory
    root_dir = html_path_split[0]
    child_dir = html_path_split[1]

    cur_html_path = os.path.join(save_dir, '{}.html'.format(root_dir))
    if os.path.exists(cur_html_path):

        fp = open(cur_html_path, 'r')
        lines_written = fp.readlines()
        fp.close()

        fp = open(cur_html_path, 'a+')
        child_path = os.path.join(root_dir, f'{child_dir}.html')
        line_to_write = f'<tr><td><a href=\"{child_path}\">{child_dir}</a></td></tr>\n'

        if line_to_write not in lines_written:
            fp.write('<html><head></head><body><table>\n')
            fp.write(line_to_write)
            fp.write('</table></body></html>')
        fp.close()

    else:

        fp = open(cur_html_path, 'w')

        child_path = os.path.join(root_dir, f'{child_dir}.html')
        line_to_write = f'<tr><td><a href=\"{child_path}\">{child_dir}</a></td></tr>\n'

        fp.write('<html><head></head><body><table>\n')
        fp.write(line_to_write)
        fp.write('</table></body></html>')

        fp.close()

    child_path = '/'.join(html_path.split('/')[1:])
    save_dir = os.path.join(save_dir, root_dir)
    create_recursive_html_link(child_path, save_dir)


def visualize_attention_maps(attention_maps_all, save_dir, width, height, tokens_vis):
    r"""Function to visualize attention maps.
    Args:
        save_dir (str): Path to save attention maps
        batch_size (int): Batch size
        sampler_order (int): Sampler order
    """

    rand_name = list(attention_maps_all.keys())[0]
    nsteps = len(attention_maps_all[rand_name])
    hw_ori = width * height

    # html_path = save_dir + '.html'
    text_input = save_dir.split('/')[-1]
    # f = open(html_path, 'wt')

    all_html_paths = []

    for step_num in range(0, nsteps, 5):

        # if cond_id == 'cond':
        #     attention_maps_cur = attention_maps_cond[step_num]
        # else:
        #     attention_maps_cur = attention_maps_uncond[step_num]

        attention_maps = dict()

        for layer in attention_maps_all.keys():

            attention_ind = attention_maps_all[layer][step_num].cpu()

            # Attention maps are of shape [batch_size, nkeys, 77]
            # since they are averaged out while collecting from hooks to save memory.
            # Now split the heads from batch dimension
            bs, hw, nclip = attention_ind.shape
            down_ratio = np.sqrt(hw_ori // hw)
            width_cur = int(width // down_ratio)
            height_cur = int(height // down_ratio)
            attention_ind = attention_ind.reshape(
                bs, height_cur, width_cur, nclip)

            attention_maps[layer] = attention_ind

        # Obtain heatmaps corresponding to random heads and individual heads

        html_names = save_attention_heatmaps(
            attention_maps, tokens_vis, save_dir=save_dir, prefix='step_{}/attention_maps_cond'.format(
                step_num)
        )

        # Write the logic for recursively creating pages
        for html_name_cur in html_names:
            all_html_paths.append(os.path.join(text_input, html_name_cur))

    save_dir_root = '/'.join(save_dir.split('/')[0:-1])
    for html_pth in all_html_paths:
        create_recursive_html_link(html_pth, save_dir_root)


def plot_attention_maps(atten_map_list, obj_tokens, save_dir, seed, tokens_vis=None):
    for i, attn_map in enumerate(atten_map_list):
        n_obj = len(attn_map)
        plt.figure()
        plt.clf()

        fig, axs = plt.subplots(
            ncols=n_obj+1, gridspec_kw=dict(width_ratios=[1 for _ in range(n_obj)]+[0.1]))

        fig.set_figheight(3)
        fig.set_figwidth(3*n_obj+0.1)

        cmap = plt.get_cmap('YlOrRd')

        vmax = 0
        vmin = 1
        for tid in range(n_obj):
            attention_map_cur = attn_map[tid]
            vmax = max(vmax, float(attention_map_cur.max()))
            vmin = min(vmin, float(attention_map_cur.min()))

        for tid in range(n_obj):
            sns.heatmap(
                attn_map[tid][0], annot=False, cbar=False, ax=axs[tid],
                cmap=cmap, vmin=vmin, vmax=vmax
            )
            axs[tid].set_axis_off()

            if tokens_vis is not None:
                if tid == n_obj-1:
                    axs_xlabel = 'other tokens'
                else:
                    axs_xlabel = ''
                    for token_id in obj_tokens[tid]:
                        axs_xlabel += ' ' + tokens_vis[token_id.item() -
                                                       1][:-len('</w>')]
                axs[tid].set_title(axs_xlabel)

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, cax=axs[-1])

        fig.tight_layout()

        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        img = np.frombuffer(canvas.tostring_rgb(),
                            dtype='uint8').reshape((height, width, 3))
        # plt.savefig(os.path.join(
        #     save_dir, 'average_seed%d_attn%d.jpg' % (seed, i)), dpi=100)
        plt.close('all')
    return img


def get_average_attention_maps(attention_maps, save_dir, width, height, obj_tokens, seed=0, tokens_vis=None,
                               preprocess=False):
    r"""Function to visualize attention maps.
    Args:
        save_dir (str): Path to save attention maps
        batch_size (int): Batch size
        sampler_order (int): Sampler order
    """

    # Split attention maps over steps
    attention_maps_cond, _ = split_attention_maps_over_steps(
        attention_maps
    )

    nsteps = len(attention_maps_cond)
    hw_ori = width * height

    attention_maps = []
    for obj_token in obj_tokens:
        attention_maps.append([])

    for step_num in range(nsteps):
        attention_maps_cur = attention_maps_cond[step_num]

        for layer in attention_maps_cur.keys():
            if step_num < 10 or layer not in CrossAttentionLayers:
                continue

            attention_ind = attention_maps_cur[layer].cpu()

            # Attention maps are of shape [batch_size, nkeys, 77]
            # since they are averaged out while collecting from hooks to save memory.
            # Now split the heads from batch dimension
            bs, hw, nclip = attention_ind.shape
            down_ratio = np.sqrt(hw_ori // hw)
            width_cur = int(width // down_ratio)
            height_cur = int(height // down_ratio)
            attention_ind = attention_ind.reshape(
                bs, height_cur, width_cur, nclip)
            for obj_id, obj_token in enumerate(obj_tokens):
                if obj_token[0] == -1:
                    attention_map_prev = torch.stack(
                        [attention_maps[i][-1] for i in range(obj_id)]).sum(0)
                    attention_maps[obj_id].append(
                        attention_map_prev.max()-attention_map_prev)
                else:
                    obj_attention_map = attention_ind[:, :, :, obj_token].max(-1, True)[
                        0].permute([3, 0, 1, 2])
                    # obj_attention_map = attention_ind[:, :, :, obj_token].mean(-1, True).permute([3, 0, 1, 2])
                    obj_attention_map = torchvision.transforms.functional.resize(obj_attention_map, (height, width),
                                                                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
                    attention_maps[obj_id].append(obj_attention_map)

    attention_maps_averaged = []
    for obj_id, obj_token in enumerate(obj_tokens):
        if obj_id == len(obj_tokens) - 1:
            attention_maps_averaged.append(
                torch.cat(attention_maps[obj_id]).mean(0))
        else:
            attention_maps_averaged.append(
                torch.cat(attention_maps[obj_id]).mean(0))

    attention_maps_averaged_normalized = []
    attention_maps_averaged_sum = torch.cat(attention_maps_averaged).sum(0)
    for obj_id, obj_token in enumerate(obj_tokens):
        attention_maps_averaged_normalized.append(
            attention_maps_averaged[obj_id]/attention_maps_averaged_sum)

    if obj_tokens[-1][0] != -1:
        attention_maps_averaged_normalized = (
            torch.cat(attention_maps_averaged)/0.001).softmax(0)
        attention_maps_averaged_normalized = [
            attention_maps_averaged_normalized[i:i+1] for i in range(attention_maps_averaged_normalized.shape[0])]

    if preprocess:
        selem = square(5)
        selem = square(3)
        selem = square(1)
        attention_maps_averaged_eroded = [erosion(skimage.img_as_float(
            map[0].numpy()*255), selem) for map in attention_maps_averaged_normalized[:2]]
        attention_maps_averaged_eroded = [(torch.from_numpy(map).unsqueeze(
            0)/255. > 0.8).float() for map in attention_maps_averaged_eroded]
        attention_maps_averaged_eroded.append(
            1 - torch.cat(attention_maps_averaged_eroded).sum(0, True))
        plot_attention_maps([attention_maps_averaged, attention_maps_averaged_normalized,
                            attention_maps_averaged_eroded], obj_tokens, save_dir, seed, tokens_vis)
        attention_maps_averaged_eroded = [attn_mask.unsqueeze(1).repeat(
            [1, 4, 1, 1]).cuda() for attn_mask in attention_maps_averaged_eroded]
        return attention_maps_averaged_eroded
    else:
        plot_attention_maps([attention_maps_averaged, attention_maps_averaged_normalized],
                            obj_tokens, save_dir, seed, tokens_vis)
        attention_maps_averaged_normalized = [attn_mask.unsqueeze(1).repeat(
            [1, 4, 1, 1]).cuda() for attn_mask in attention_maps_averaged_normalized]
        return attention_maps_averaged_normalized


def get_average_attention_maps_threshold(attention_maps, save_dir, width, height, obj_tokens, seed=0, threshold=0.02):
    r"""Function to visualize attention maps.
    Args:
        save_dir (str): Path to save attention maps
        batch_size (int): Batch size
        sampler_order (int): Sampler order
    """

    _EPS = 1e-8
    # Split attention maps over steps
    attention_maps_cond, _ = split_attention_maps_over_steps(
        attention_maps
    )

    nsteps = len(attention_maps_cond)
    hw_ori = width * height

    attention_maps = []
    for obj_token in obj_tokens:
        attention_maps.append([])

    # for each side prompt, get attention maps for all steps and all layers
    for step_num in range(nsteps):
        attention_maps_cur = attention_maps_cond[step_num]
        for layer in attention_maps_cur.keys():
            attention_ind = attention_maps_cur[layer].cpu()
            bs, hw, nclip = attention_ind.shape
            down_ratio = np.sqrt(hw_ori // hw)
            width_cur = int(width // down_ratio)
            height_cur = int(height // down_ratio)
            attention_ind = attention_ind.reshape(
                bs, height_cur, width_cur, nclip)
            for obj_id, obj_token in enumerate(obj_tokens):
                if attention_ind.shape[1] > width//2:
                    continue
                if obj_token[0] != -1:
                    obj_attention_map = attention_ind[:, :, :,
                                                      obj_token].mean(-1, True).permute([3, 0, 1, 2])
                    obj_attention_map = torchvision.transforms.functional.resize(obj_attention_map, (height, width),
                                                                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
                    attention_maps[obj_id].append(obj_attention_map)

    # average of all steps and layers, thresholding
    attention_maps_thres = []
    attention_maps_averaged = []
    for obj_id, obj_token in enumerate(obj_tokens):
        if obj_token[0] != -1:
            average_map = torch.cat(attention_maps[obj_id]).mean(0)
            attention_maps_averaged.append(average_map)
            attention_maps_thres.append((average_map > threshold).float())

    # get the remaining region except for the original prompt
    attention_maps_averaged_normalized = []
    attention_maps_averaged_sum = torch.cat(attention_maps_thres).sum(0) + _EPS
    for obj_id, obj_token in enumerate(obj_tokens):
        if obj_token[0] != -1:
            attention_maps_averaged_normalized.append(
                attention_maps_thres[obj_id]/attention_maps_averaged_sum)
        else:
            attention_map_prev = torch.stack(
                attention_maps_averaged_normalized).sum(0)
            attention_maps_averaged_normalized.append(1.-attention_map_prev)

    plot_attention_maps(
        [attention_maps_averaged, attention_maps_averaged_normalized], save_dir, seed)

    attention_maps_averaged_normalized = [attn_mask.unsqueeze(1).repeat(
        [1, 4, 1, 1]).cuda() for attn_mask in attention_maps_averaged_normalized]
    # attention_maps_averaged_normalized = attention_maps_averaged_normalized.unsqueeze(1).repeat([1, 4, 1, 1]).cuda()
    return attention_maps_averaged_normalized


def get_token_maps(selfattn_maps, crossattn_maps, n_maps, save_dir, width, height, obj_tokens, kmeans_seed=0, tokens_vis=None,
                   preprocess=False, segment_threshold=0.3, num_segments=5, return_vis=False, save_attn=False):
    r"""Function to visualize attention maps.
    Args:
        save_dir (str): Path to save attention maps
        batch_size (int): Batch size
        sampler_order (int): Sampler order
    """

    target_height = np.ceil(height / 4).astype(int)
    target_width = np.ceil(width / 4).astype(int)
    target_hw = target_height * target_width
    hw = width * height
    # attn_maps_1024 = [attn_map for attn_map in selfattn_maps.values(
    # ) if attn_map.shape[1] == resolution**2]
    # attn_maps_1024 = torch.cat(attn_maps_1024).mean(0).cpu().numpy()
    attn_maps_1024 = {target_hw: []}
    for attn_map in selfattn_maps.values():
        resolution_map = attn_map.shape[1]
        if resolution_map != target_hw:
            continue
        # attn_map = torch.nn.functional.interpolate(rearrange(attn_map, '1 c (h w) -> 1 c h w', h=resolution_map), (resolution, resolution),
        #                                            mode='bicubic', antialias=True)
        # attn_map = rearrange(attn_map, '1 (h w) a b -> 1 (a b) h w', h=resolution_map)
        attn_map = attn_map.reshape(
            1, target_height, target_width, resolution_map).permute([3, 0, 1, 2]).float()
        attn_map = torch.nn.functional.interpolate(attn_map, (target_height, target_width),
                                                   mode='bicubic', antialias=True)
        attn_maps_1024[resolution_map].append(attn_map.permute([1, 2, 3, 0]).reshape(
            1, target_hw, resolution_map))
    attn_maps_1024 = torch.cat([torch.cat(v).mean(0).cpu()
                                for v in attn_maps_1024.values() if len(v) > 0], -1).numpy()
    if save_attn:
        print('saving self-attention maps...', attn_maps_1024.shape)
        torch.save(torch.from_numpy(attn_maps_1024),
                   'results/maps/selfattn_maps.pth')
    seed_everything(kmeans_seed)
    # import ipdb;ipdb.set_trace()
    # kmeans = KMeans(n_clusters=num_segments,
    #                 n_init=10).fit(attn_maps_1024)
    # clusters = kmeans.labels_
    # clusters = clusters.reshape(resolution, resolution)
    # mesh = np.array(np.meshgrid(range(resolution), range(resolution), indexing='ij'), dtype=np.float32)/resolution
    # dists = mesh.reshape(2, -1).T
    # delta = 0.01
    # spatial_sim = rbf_kernel(dists, dists)*delta
    sc = SpectralClustering(num_segments, affinity='precomputed', n_init=100,
                            assign_labels='kmeans')
    clusters = sc.fit_predict(attn_maps_1024)
    clusters = clusters.reshape(target_height, target_width)
    fig = plt.figure()
    plt.imshow(clusters)
    plt.axis('off')
    # plt.savefig(os.path.join(save_dir, 'segmentation_k%d_seed%d.jpg' % (num_segments, kmeans_seed)),
    #             bbox_inches='tight', pad_inches=0)
    if return_vis:
        canvas = fig.canvas
        canvas.draw()
        cav_width, cav_height = canvas.get_width_height()
        segments_vis = np.frombuffer(canvas.tostring_rgb(),
                                     dtype='uint8').reshape((cav_height, cav_width, 3))

    plt.close()

    # label the segmentation mask using cross-attention maps
    cross_attn_maps_1024 = []
    for attn_map in crossattn_maps.values():
        resolution_map = np.sqrt(attn_map.shape[1]).astype(int)
        # if resolution_map != 16:
        # continue
        attn_map = attn_map.reshape(
            1, resolution_map, resolution_map, -1).permute([0, 3, 1, 2]).float()
        attn_map = torch.nn.functional.interpolate(attn_map, (target_height, target_width),
                                                   mode='bicubic', antialias=True)
        cross_attn_maps_1024.append(attn_map.permute([0, 2, 3, 1]))

    cross_attn_maps_1024 = torch.cat(
        cross_attn_maps_1024).mean(0).cpu().numpy()
    normalized_span_maps = []
    for token_ids in obj_tokens:
        token_ids = torch.clip(token_ids, 0, 76)
        span_token_maps = cross_attn_maps_1024[:, :, token_ids.numpy()]
        normalized_span_map = np.zeros_like(span_token_maps)
        for i in range(span_token_maps.shape[-1]):
            curr_noun_map = span_token_maps[:, :, i]
            normalized_span_map[:, :, i] = (
                # curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
                curr_noun_map - np.abs(curr_noun_map.min())) / (curr_noun_map.max()-curr_noun_map.min())
        normalized_span_maps.append(normalized_span_map)
    foreground_token_maps = [np.zeros([clusters.shape[0], clusters.shape[1]]).squeeze(
    ) for normalized_span_map in normalized_span_maps]
    background_map = np.zeros([clusters.shape[0], clusters.shape[1]]).squeeze()
    for c in range(num_segments):
        cluster_mask = np.zeros_like(clusters)
        cluster_mask[clusters == c] = 1.
        is_foreground = False
        for normalized_span_map, foreground_nouns_map, token_ids in zip(normalized_span_maps, foreground_token_maps, obj_tokens):
            score_maps = [cluster_mask * normalized_span_map[:, :, i]
                          for i in range(len(token_ids))]
            scores = [score_map.sum() / cluster_mask.sum()
                      for score_map in score_maps]
            if max(scores) > segment_threshold:
                foreground_nouns_map += cluster_mask
                is_foreground = True
        if not is_foreground:
            background_map += cluster_mask
    foreground_token_maps.append(background_map)

    # resize the token maps and visualization
    resized_token_maps = torch.cat([torch.nn.functional.interpolate(torch.from_numpy(token_map).unsqueeze(0).unsqueeze(
        0), (height, width), mode='bicubic', antialias=True)[0] for token_map in foreground_token_maps]).clamp(0, 1)

    resized_token_maps = resized_token_maps / \
        (resized_token_maps.sum(0, True)+1e-8)
    resized_token_maps = [token_map.unsqueeze(
        0) for token_map in resized_token_maps]
    foreground_token_maps = [token_map[None, :, :]
                             for token_map in foreground_token_maps]
    if preprocess:
        selem = square(5)
        eroded_token_maps = torch.stack([torch.from_numpy(erosion(skimage.img_as_float(
            map[0].numpy()*255), selem))/255. for map in resized_token_maps[:-1]]).clamp(0, 1)
        # import ipdb; ipdb.set_trace()
        eroded_background_maps = (1-eroded_token_maps.sum(0, True)).clamp(0, 1)
        eroded_token_maps = torch.cat([eroded_token_maps, eroded_background_maps])
        eroded_token_maps = eroded_token_maps / (eroded_token_maps.sum(0, True)+1e-8)
        resized_token_maps = [token_map.unsqueeze(
            0) for token_map in eroded_token_maps]

    token_maps_vis = plot_attention_maps([foreground_token_maps, resized_token_maps], obj_tokens,
                                         save_dir, kmeans_seed, tokens_vis)
    resized_token_maps = [token_map.unsqueeze(1).repeat(
        [1, 4, 1, 1]).to(attn_map.dtype).cuda() for token_map in resized_token_maps]
    if return_vis:
        return resized_token_maps, segments_vis, token_maps_vis
    else:
        return resized_token_maps
