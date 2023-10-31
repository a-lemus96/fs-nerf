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
            '--model', choices=['nerf', 'sinerf'], default='nerf',
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
            '--dataset', default='synthetic', 
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
            '--white_bkgd', action="store_true",
            help="Use white backgroung for training imgs"
    )
    parser.add_argument(
            '--img_mode', action="store_true",
            help="If set, iterate over images instead of rays for training")

    #-------------------------------training-----------------------------------#

    parser.add_argument(
            '--n_iters', default=20**3, type=int,
            help='Number of training iterations'
    )
    parser.add_argument(
            '--warmup_iters', default=2500, type=int, 
            help="Number of iterations for mip scheduler's warmup phase"
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
            '--lrf', default=5e-5, type=float,
            help='Final learning rate for optimizer'
    )
    parser.add_argument(
            '--decay_rate', default=0.999, type=float,
            help='Decay rate for exponential learning rate scheduler'
    )
    parser.add_argument(
            '--T_lr', default=20**3, type=int,
            help='Number of iterations for learning rate decay'
    )
    parser.add_argument(
            '--scheduler', choices=['const', 'exp', 'proot'], default='exp',
            help='Learning rate scheduler'
    )
    parser.add_argument(
            '--device_num', default=0, type=int,
            help="Number of CUDA device to be used for training"
    )

    #-------------------------------validation---------------------------------#

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
            '--p', default=2, type=int,
            help='p-root of the alpha regularizer scheduler'
    )
    parser.add_argument(
            '--T', default=5000, type=int,
            help='Number of iterations for active alpha regularizer scheduler'
    )
    parser.add_argument(
            '--reg', choices=['l1', 'l2'], default='l2',
            help='Norm for penalizing model parameters'
    )

    #---------------------------------depth------------------------------------#

    parser.add_argument(
            '--use_bkgd', action="store_true",
            help='If set, use background pixels for depth supervision'
    )
    parser.add_argument(
            '--mu', default=None, type=float,
            help='Balancing hyperparameter for depth loss'
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
