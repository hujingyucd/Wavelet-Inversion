import pywt
import torch

debug_base_folder = r"/home/jyhu/condiff/control_diff_release/debug/"
backup_path = None


## training
# data
use_preload = True

data_files = []


num_points =  4096
interval = 1
test_nums = 16
batch_num_points = 64 ** 3
first_k = None
use_surface_samples = False
sample_resolution = 64
sample_ratio = 0.8
load_ram = False
loss_function = torch.nn.MSELoss()
new_low_fix = True
remove_reductant = True
mix_precision = True

#
batch_size = 8
lr = 5e-5
lr_decay = False
lr_decay_feq = 500
lr_decay_rate = 0.998
vis_results = True
progressive = True
data_worker = 25
beta1 = 0.9
beta2 = 0.999
optimizer = torch.optim.Adam

## network
resolution = 256
latent_dim = 256
padding_mode = 'zero'
wavelet_type = 'bior6.8'
wavelet = pywt.Wavelet(wavelet_type)
max_depth = pywt.dwt_max_level(data_len = resolution, filter_len=wavelet.dec_len)
latent_dim = int(resolution // (max_depth+1) * (max_depth+1)) # round down
activation = torch.nn.LeakyReLU(0.02) #torch.nn.LeakyReLU(0.02)
use_fourier_features = True
fourier_norm = 1.0
use_dense_conv = True
linear_layers = [128, 128, 128, 128]
code_bound = 0.1
weight_sigma = 0.02
scale_coordinates = True
train_only_current_level = True
lr_weight_decay_after_stage = False
lr_decay_rate_after_stage = 0.01
use_clip = False
clip_half = False
clip_bound = 0.1
train_low_stage_with_full = False
use_gradient_clip = False
gradient_clip_value = 1.0
use_instance_norm = True
use_instance_affine = True
use_layer_norm = False
use_layer_affine = False
training_stage = max_depth
train_with_gt_coeff = True
zero_stages = []
gt_stages = [0,1,2]

### diffusion setting
from models.module.gaussian_diffusion import  ModelMeanType, ModelVarType, LossType
use_diffusion = True
diffusion_step = 1000
diffusion_model_var_type = ModelVarType.FIXED_SMALL
diffusion_learn_sigma = False
diffusion_sampler = 'second-order'
diffusion_model_mean_type = ModelMeanType.EPSILON
diffusion_rescale_timestep = False
diffusion_loss_type = LossType.MSE
diffusion_beta_schedule = 'linear'
diffusion_scale_ratio = 1.0
unet_model_channels = 64
unet_num_res_blocks = 3
unet_channel_mult = (1, 1, 2, 4)
unet_channel_mult_low = (1, 2, 2, 2)
unet_activation = None #torch.nn.LeakyReLU(0.1)
attention_resolutions = []
if diffusion_learn_sigma:
    diffusion_model_var_type = ModelVarType.LEARNED_RANGE
    diffusion_loss_type = LossType.RESCALED_MSE

###
highs_use_conv3d = False
conv3d_tuple_layers_highs_append = [
                       (8, (5, 5, 5), (2, 2, 2)),
                       (8, (5, 5, 5), (1, 1, 1)),
                       ]

###
highs_use_downsample_features = False
highs_use_unent = False
downsample_features_dim = 64
conv3d_downsample_tuple_layers = [
    (16, (3, 3, 3), (2, 2, 2)),
    (16, (3, 3, 3), (1, 1, 1)),
    (32, (3, 3, 3), (2, 2, 2)),
    (32, (3, 3, 3), (1, 1, 1)),
    (64, (3, 3, 3), (2, 2, 2)),
    (64, (3, 3, 3), (1, 1, 1)),
    (128, (3, 3, 3), (2, 2, 2)),
    (128, (3, 3, 3), (1, 1, 1)),
]

# use discriminator
use_discriminator = False
discriminator_weight = 0.01
i_dim = 2
z_dim = 1
d_dim = 16

## resume
starting_epoch = 0
training_epochs = 3000
saving_intervals = 20
starting_stage = max_depth
special_symbol = ''
network_resume_path = None 
optimizer_resume_path = None
discriminator_resume_path = None
discriminator_opt_resume_path = None
exp_idx = 15

## for voxel_ae
lr_ae = 1e-5
ae_input_channel = 1
ae_z_dim = 256
ae_ef_dim = 32
scale_factor = 2
sample_mode = 'nearest'
negative_slope = 0.02
use_trans3d = True
use_aeConv3d = False
upsampleconv3d_tuple_layers = [
                       # (128, (3, 3, 3), (1, 1, 1)),
                       (64, (3, 3, 3), (2, 2, 2)),
                       #(128, (3, 3, 3), (1, 1, 1)),
                       #(128, (3, 3, 3), (1, 1, 1)),
                       (64, (3, 3, 3), (2, 2, 2)),
                       #(128, (3, 3, 3), (1, 1, 1)),
                       #(128, (3, 3, 3), (1, 1, 1)),
                       (32, (3, 3, 3), (2, 2, 2)),
                       (32, (3, 3, 3), (1, 1, 1)),
                       #(128, (3, 3, 3), (1, 1, 1)),
                       (16, (5, 5, 5), (2.5, 2.5, 2.5)),
                       (16, (5, 5, 5), (1, 1, 1)),
                       #(128, (5, 5, 5), (1, 1, 1)),
                       #(128, (5, 5, 5), (2, 2, 2)),
                       #(128, (5, 5, 5), (1, 1, 1)),
                       #(128, (5, 5, 5), (1, 1, 1)),
                       ]
use_latent_rep = False
pretrained_ae_path = None
mix_precision = True
join_training = True
use_pretrained = True
