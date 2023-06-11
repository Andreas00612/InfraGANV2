import torch
import math
import matplotlib.pyplot  as plt
import numpy as np
def feature_visualization(x, module_type, stage, n=32, save_dir=r'D:\InfraGANV2\visualize_GAN'):
    """
    Visualize feature maps of a given model module during inference.
    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """

    batch, channels, height, width = x.shape # batch, channels, height, width
    #x = x.cpu().detach().numpy()
    if height > 1 and width > 1:
        f = save_dir + '/' +  f"stage{stage}_{module_type}_features.png"  # filename

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(n, channels)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            blocks_ = blocks[i].detach().numpy()
            ax[i].imshow(blocks_.squeeze())  # cmap='gray'
            ax[i].axis('off')

        #LOGGER.info(f'Saving {f}... ({n}/{channels})')
        print((f'Saving {f}... ({n}/{channels})'))
        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close()

