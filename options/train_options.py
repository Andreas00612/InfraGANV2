from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        # self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        # self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        # self.parser.add_argument('--lambda_identity', type=float, default=0.5,
        #                          help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
        #                               'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50,
                                 help='multiply by a gamma every lr_decay_iters iterations')

        self.parser.add_argument('--n_Gen', type=int, default=2)
        self.parser.add_argument('--D_input', type=str, default='cat')

        #self.parser.add_argument('--drop_rate', type=float, default=-1)
        # self.parser.add_argument('--n_layers_D',type=int ,default=4)

        #### add loss function ####

        self.parser.add_argument('--loss_l1', action='store_true')
        self.parser.add_argument('--lambda_l1', type=float, default=50.0)

        self.parser.add_argument('--loss_lpips', action='store_true')
        self.parser.add_argument('--lambda_lpips', type=float, default=50.0)

        self.parser.add_argument('--loss_monce', action='store_true')
        self.parser.add_argument('--lambda_monce', type=float, default=10.0)

        self.parser.add_argument('--loss_identity', action='store_true')
        self.parser.add_argument('--lambda_id', type=float, default=10.0)

        self.parser.add_argument('--loss_sobel', action='store_true')
        self.parser.add_argument('--lambda_sobel', type=float, default=10.0)

        self.parser.add_argument('--loss_ssim', type=str, default=None)
        self.parser.add_argument('--lambda_ssim', type=float, default=50.0)

        self.parser.add_argument('--loss_perceptual', action='store_true')
        self.parser.add_argument('--lambda_perceptual', type=float, default=10.0)

        self.parser.add_argument('--loss_mse', action='store_true')
        self.parser.add_argument('--lambda_mse', type=float, default=50.0)

        self.parser.add_argument('--loss_CCP', action='store_true')
        self.parser.add_argument('--lambda_CCP', type=float, default=5.0)
        self.parser.add_argument('--num_s', type=int, default=8, help='number of sampled anchor vectors')
        self.parser.add_argument('--num_l', type=int, default=3, help='number of layers to calculate CCPL')

        self.parser.add_argument('--loss_tv', action='store_true')
        self.parser.add_argument('--lambda_tv', type=float, default=10.0)


        self.isTrain = True


