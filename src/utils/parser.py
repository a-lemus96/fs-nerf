# stdlib imports
import argparse

def config_parser() -> argparse.Namespace:
    """Creates a parser for command-line arguments.
    ----------------------------------------------------------------------------
    """
    parser = argparse.ArgumentParser(description='Train NeRF for view synthesis.')

    # Encoder(s)
    parser.add_argument('--d_input', dest='d_input', default=3, type=int,
                        help='Spatial input dimension')
    parser.add_argument('--n_freqs', dest='n_freqs', default=10, type=int,
                        help='Number of encoding functions for spatial coords')
    parser.add_argument('--log_space', dest='log_space', action="store_false",
                        help='If not set, frecuency scale in log space')
    parser.add_argument('--no_dirs', dest='no_dirs', action="store_true",
                        help='If set, do not model view-dependent effects')
    parser.add_argument('--n_freqs_views', dest='n_freqs_views', default=4,
                        type=int, help='Number of encoding fns for view dirs')

    # Model(s)
    parser.add_argument('--d_filter', dest='d_filter', default=256, type=int,
                        help='Linear layer filter dimension')
    parser.add_argument('--n_layers', dest='n_layers', default=8, type=int,
                        help='Number of layers preceding bottleneck')
    parser.add_argument('--skip', dest='skip', default=[4], type=list,
                        help='Layers at which to apply input residual')
    parser.add_argument('--use_fine', dest='use_fine', action="store_true",
                        help='Creates and uses fine NeRF model')
    parser.add_argument('--d_filter_fine', dest='d_filter_fine', default=256,
                        type=int, help='Linear layer filter dim for fine model')
    parser.add_argument('--n_layers_fine', dest='n_layers_fine', default=8,
                        type=int, help='Number of fine layers before bottleneck')

    # Stratified sampling
    parser.add_argument('--n_samples', dest='n_samples', default=64, type=int,
                        help='Number of stratified samples per ray')
    parser.add_argument('--perturb', dest='perturb', action="store_false",
                        help='If set, do not apply noise to spatial coords')
    parser.add_argument('--inv_depth', dest='inv_depth', action="store_true",
                        help='If set, sample points linearly in inverse depth')

    # Hierarchical sampling
    parser.add_argument('--n_samples_hierch', dest='n_samples_hierch', default=128,
                        type=int, help='Number of hierarchical samples per ray')
    parser.add_argument('--perturb_hierch', dest='perturb_hierch',
                        action="store_false", 
                        help='Applies noise to hierarchical samples')

    # Dataset
    parser.add_argument('--dataset', dest='dataset', default='synthetic', 
                        type=str, help="Dataset to choose scenes from")
    parser.add_argument('--scene', dest='scene', default='lego', type=str,
                        help="Scene to be used for training")
    parser.add_argument('--n_imgs', dest='n_imgs', default=100, type=int,
                        help="Number of images to be used for training")
    parser.add_argument('--white_bkgd', dest='white_bkgd', default=False,
                        type=bool, help="Use white backgroung for training imgs")

    # optimization
    parser.add_argument('--lrate', dest='lrate', default=5e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--mu', dest='mu', default=None, type=float,
                        help='Balancing hyperparameter for depth loss')

    # training 
    parser.add_argument('--n_iters', dest='n_iters', default=10**5, type=int,
                        help='Number of training iterations')
    parser.add_argument('--warmup_iters', dest='warmup_iters', default=2500,
                        type=int, help='Number of iterations for warmup phase')
    parser.add_argument('--batch_size', dest='batch_size', default=2**12, type=int,
                        help='Number of rays per optimization step')
    parser.add_argument('--device_num', dest='device_num', default=1, type=int,
                        help="Number of CUDA device to be used for training")
    parser.add_argument('--nerfacc', dest='nerfacc', action="store_true",
                        help='If set, use NeRF-Acc rendering')

    # validation
    parser.add_argument('--display_rate', dest='display_rate', default=1000, type=int,
                        help='Display rate for test output measured in iterations')
    parser.add_argument('--val_rate', dest='val_rate', default=100, type=int,
                        help='Test image evaluation rate')

    # directories
    parser.add_argument('--out_dir', dest='out_dir', default="../out/",
                        type=str, help="Base directory for storing results")

    # video Rendering
    parser.add_argument('--render_only', dest='render_only', action='store_true',
                        help='If set, load pretrained model to render video')

    # debugging
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='If set, run in debug mode')
            

    args = parser.parse_args()

    return args
