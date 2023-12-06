# stdlib imports
import argparse

def config_parser() -> argparse.Namespace:
    """Creates a parser for command-line arguments.
    ----------------------------------------------------------------------------
    """
    parser = argparse.ArgumentParser(
            description='Train NeRF for view synthesis.'
    )

    #-------------------------------encoder------------------------------------#


    parser.add_argument(
            '--d_input', default=3, type=int,
            help='Spatial input dimension'
    )
    parser.add_argument(
            '--n_freqs', default=10, type=int,
            help='Number of encoding functions for spatial coords'
    )
    parser.add_argument(
            '--log_space', action="store_false",
            help='If not set, frecuency scale in log space'
    )
    parser.add_argument(
            '--no_dirs', action="store_true",
            help='If set, do not model view-dependent effects'
    )
    parser.add_argument(
            '--n_freqs_views', dest='n_freqs_views', default=4,
            type=int, help='Number of encoding fns for view dirs'
    )

    #--------------------------------model-------------------------------------#

    parser.add_argument(
            '--model', choices=['nerf', 'sinerf', 'ensemble'], default='nerf',
            help='Model to be used for training'
    )
    parser.add_argument(
            '--d_filter', default=256, type=int,
            help='Linear layer filter dimension'
    )
    parser.add_argument(
            '--n_layers', default=8, type=int,
            help='Number of layers preceding bottleneck'
    )
    parser.add_argument(
            '--skip', default=[4], type=list,
            help='Layers at which to apply input residual for NeRF'
    )

    #---------------------------------data-------------------------------------#

    parser.add_argument(
            '--dataset', choices=['synthetic', 'llff'], default='synthetic', 
            type=str, help="Dataset to choose scenes from"
    )
    parser.add_argument(
            '--scene', default='lego', type=str,
            help="Scene to be used for training"
    )
    parser.add_argument(
            '--n_imgs', default=100, type=int,
            help="Number of images to be used for training"
    )
    parser.add_argument(
            '--img_mode', action="store_true",
            help="If set, iterate over images instead of rays for training")

    # args for synthetic dataset
    parser.add_argument(
            '--white_bkgd', action="store_true",
            help="Use white background for training imgs"
    )

    # args for llff dataset
    parser.add_argument(
            '--factor', default=4, type=int,
            help="Downsample factor for LLFF dataset"
    )
    parser.add_argument(
            '--bd_factor', default=0.75, type=float,
            help="Bound factor for LLFF dataset"
    )
    parser.add_argument(
            '--no_ndc', action="store_true",
            help="If set, do not use normalized device coordinates"
    )
    parser.add_argument(
            '--no_recenter', action="store_true",
            help="If set, do not recenter LLFF dataset"
    )

    #-------------------------------training-----------------------------------#

    parser.add_argument(
            '--n_iters', default=20**3, type=int,
            help='Number of training iterations'
    )
    parser.add_argument(
            '--batch_size', default=1024, type=int,
            help='Number of rays per optimization step'
    )
    parser.add_argument(
            '--lro', default=5e-4, type=float,
            help='Initial learning rate for optimizer'
    )
    parser.add_argument(
            '--lrf', default=0., type=float,
            help='Final learning rate for optimizer'
    )
    parser.add_argument(
            '--decay_rate', default=0.1, type=float,
            help='Decay rate for exponential learning rate scheduler'
    )
    parser.add_argument(
            '--Td', default=250000, type=int,
            help='Number of iterations for learning rate decay'
    )
    parser.add_argument(
            '--scheduler', choices=['const', 'exp'], default='exp',
            help='Learning rate scheduler'
    )

    #-------------------------------validation---------------------------------#

    parser.add_argument(
            '--no_val', action='store_true',
            help='If set, do not perform validation during training'
    )
    parser.add_argument(
            '--val_rate', default=500, type=int,
            help='Number of iterations between validation steps'
    )
    parser.add_argument(
            '--val_ratio', default=0.25, type=float,
            help='Ratio of val data to be used in between epochs'
    )

    #-----------------------------regularizers---------------------------------#

    parser.add_argument(
            '--ao', default=None, type=float,
            help='Initial alpha value for regularizing model parameter weights'
    )
    parser.add_argument(
            '--reg_ratio', default=0.5, type=float,
            help='Ratio of iterations for alpha regularizer scheduler'
    )
    parser.add_argument(
            '--p', default=2, type=int,
            help='p-root of the alpha regularizer scheduler'
    )
    parser.add_argument(
            '--reg', choices=['l1', 'l2'], default='l1',
            help='Norm for penalizing model parameters'
    )
    parser.add_argument(
            '--M', default=15, type=int,
            help='Occlusion regularization range parameter'
    )
    parser.add_argument(
            '--beta', default=None, type=float,
            help='Occlusion regularization importance parameter'
    )
    #-------------------------------logging------------------------------------#

    parser.add_argument(
            '--out_dir', default="../out/",
            type=str, help="Base directory for storing results")

    #-------------------------------debugging----------------------------------#

    parser.add_argument(
            '--debug', action='store_true',
            help='If set, run in debug mode'
    )

    #----------------------------videorendering--------------------------------#

    parser.add_argument(
            '--render_only', action='store_true',
            help='If set, load pretrained model to render video'
    )

    args = parser.parse_args()

    return args
