import os
import importlib
import torch
import numpy as np
import mcubes
import torch.nn.functional as F
from data.data import SDFSamples
from models.network import MultiScaleMLP,SparseComposer, create_coordinates
from models.module.dwt import DWTInverse3d_Laplacian, DWTForward3d_Laplacian
from models.module.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from models.module.respace import SpacedDiffusion, space_timesteps
from utils.debugger import MyDebugger
from models.module.resample import UniformSampler, LossSecondMomentResampler, LossAwareSampler
import time
from models.module.diffusion_network import UNetModel, MyUNetModel ,Con_UNetModel, Con_ResBlock, TimeConstepEmbedSequential
from sklearn.decomposition import FastICA, PCA, IncrementalPCA, MiniBatchSparsePCA, SparsePCA, KernelPCA
from models.network_ae import VoxelEncoder, VoxelDecoder, VoxelAutoEncoder
import copy
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def process_state_dict(network_state_dict):
    for key, item in list(network_state_dict.items()):
        if 'module.' in key:
            new_key = key.replace('module.', '')
            network_state_dict[new_key] = item
            del network_state_dict[key]

    return network_state_dict

if __name__ == '__main__':
    testing_folder = r"./testing_folder/"


    detail_path = r"./testing_folder/model_epoch_240.pth"

    config_path = os.path.join(testing_folder, 'config.py')


    ## import config here
    spec = importlib.util.spec_from_file_location('*', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    scale_ratio = 1.0

    epoch = 900 # chair 3
    stage = 3

    config.diffusion_step = 1000
    ### debugger
    from configs import config as current_config
    debugger = MyDebugger(f'Shape_Inversion_{epoch}',
                          is_save_print_to_file=False)


    use_preload = True
    loading_files = [(r"03001627_0.1_bior6.8_3_zero_testing.npy", 3),
                     (r"03001627_0.1_bior6.8_2_zero_testing.npy", 2)]



    network_path = os.path.join(testing_folder, f'model_epoch_3_{epoch}.pth')
    ae_network_path = os.path.join(testing_folder, f'ae_model_epoch_3_{epoch}.pth')
    ### create dataset
    samples = SDFSamples(data_files = loading_files)
    ### level_indices_remap
    level_map = {idx: level for idx, (_, level) in enumerate(loading_files)}


    ### initialize network
    dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J=config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
    dwt_forward_3d_lap = DWTForward3d_Laplacian(J = config.max_depth, wave=config.wavelet, mode=config.padding_mode).to(device)
    composer_parms = dwt_inverse_3d_lap if config.use_dense_conv else None
    dwt_sparse_composer = SparseComposer(input_shape=[config.resolution, config.resolution, config.resolution],
                                         J=config.max_depth,
                                         wave=config.wavelet, mode=config.padding_mode,
                                         inverse_dwt_module=composer_parms).to(device)
    network = Con_UNetModel(in_channels=1,
                      model_channels=config.unet_model_channels,
                      out_channels=2 if hasattr(config,
                                                'diffusion_learn_sigma') and config.diffusion_learn_sigma else 1,
                      num_res_blocks=config.unet_num_res_blocks,
                      channel_mult=config.unet_channel_mult_low,
                      attention_resolutions=config.attention_resolutions,
                      use_scale_shift_norm = True,
                      dropout=0,
                      dims=3,
                      activation=config.unet_activation if hasattr(config, 'unet_activation') else None)
    ae_network = VoxelAutoEncoder(config)


    detail_network = MyUNetModel(in_channels=1,
                          spatial_size=dwt_sparse_composer.shape_list[2][0],
                          model_channels=config.unet_model_channels,
                          out_channels=1,
                          num_res_blocks=config.unet_num_res_blocks,
                          channel_mult=config.unet_channel_mult,
                          attention_resolutions=config.attention_resolutions,
                          dropout=0,
                          dims=3)

    betas = get_named_beta_schedule(config.diffusion_beta_schedule, config.diffusion_step,
                                    config.diffusion_scale_ratio)
    diffusion_module = GaussianDiffusion(betas=betas,
                                         model_var_type=config.diffusion_model_var_type,
                                         model_mean_type=config.diffusion_model_mean_type,
                                         loss_type=config.diffusion_loss_type,
                                         rescale_timesteps=config.diffusion_rescale_timestep if hasattr(
                                             config, 'diffusion_rescale_timestep') else False)
    sampler = LossSecondMomentResampler(diffusion_module)
    mse_fuction = config.loss_function


    print(f"detail predictor load from {detail_path}")
    detail_network_state_dict = process_state_dict(torch.load(detail_path))
    detail_network.load_state_dict(detail_network_state_dict)
    detail_network.to(device)
    detail_network.eval()


    print(f"Diffusion network reload model from:{network_path}")
    network_state_dict = torch.load(network_path)
    network_state_dict = process_state_dict(network_state_dict)
    network.load_state_dict(network_state_dict)
    network = network.to(device)
    network.eval()
    print(F"AE network reload model from:{ae_network_path}")
    ae_network_state_dict = torch.load(ae_network_path)
    ae_network_state_dict = process_state_dict(ae_network_state_dict)
    ae_network.load_state_dict(ae_network_state_dict)
    ae_network = ae_network.to(device)
    ae_network.eval()

    ###
    zero_stages = []
    gt_stages = [0, 1, 2]
    test_index = [ i for i in range(config.max_depth + 1) if i not in zero_stages and i not in gt_stages]
    assert len(test_index) == 1
    test_index = test_index[0]
    clip_noise = False
    evaluate_gt = False
    fixed_noise = False
    need_gt = True
    use_ddim = True
    use_interpolation = True
    respacing = [config.diffusion_step]
    noise_path = None
    use_detail = True
    ddim_eta = 1.0


    betas = get_named_beta_schedule(config.diffusion_beta_schedule, config.diffusion_step,
                                    scale_ratio)


    diffusion_module = SpacedDiffusion(use_timesteps=space_timesteps(config.diffusion_step, respacing),
                                     betas=betas,
                                     model_var_type=config.diffusion_model_var_type,
                                     model_mean_type=config.diffusion_model_mean_type,
                                     loss_type=config.diffusion_loss_type)



    testing_cnt = 1
    testing_indices = []
    for i in range(len(samples)):
        testing_indices = testing_indices + [i+1] * testing_cnt

    if fixed_noise:
        if noise_path is not None:
            noise = torch.load(noise_path).to(device)
        else:
            noise = torch.randn([1, 1] + dwt_sparse_composer.shape_list[test_index]).to(device)
    else:
        noise = None




    for m in range(int(len(testing_indices))):
        testing_sample_index = testing_indices[m]
        data = samples[testing_sample_index][0]
        low_lap, highs_lap = None, [None] * config.max_depth
        coeff_gt = data
        for j, gt in enumerate(coeff_gt):
            level = level_map[j]
            if level == config.max_depth:
                low_lap = torch.from_numpy(coeff_gt[j]).unsqueeze(0).unsqueeze(1).to(device)
            else:
                highs_lap[level] = torch.from_numpy(coeff_gt[j]).unsqueeze(0).unsqueeze(1).to(device)


        print(f"Diffusion network reload model from:{network_path}")
        network_state_dict = torch.load(network_path)
        network_state_dict = process_state_dict(network_state_dict)
        network.load_state_dict(network_state_dict)
        network = network.to(device)
        network.eval()

        num_iterations = 500
        fintune_lr = 5e-2
        ae_input = F.pad(low_lap, (9, 9, 9, 9, 9, 9), "constant", 0)
        _, z_sem = ae_network(F.pad(low_lap, (9, 9, 9, 9, 9, 9), "constant", 0))
        z_sem = torch.nn.Parameter(z_sem)
        z_sem.requires_grad = True
        features = {"input_block": [], "middle_block": [], "output_block": []}
        for module in network.input_blocks:
            if isinstance(module, TimeConstepEmbedSequential):
                if isinstance(module[0], Con_ResBlock):
                    features["input_block"].append(module[0].linear_sem(z_sem[:,:, 0,0,0]))
        for module in network.middle_block:
            if isinstance(module, Con_ResBlock):
                features["middle_block"].append(module.linear_sem(z_sem[:, :, 0, 0, 0]))
        for module in network.output_blocks:
            if isinstance(module, TimeConstepEmbedSequential):
                if isinstance(module[0], Con_ResBlock):
                    features["output_block"].append(module[0].linear_sem(z_sem[:, :, 0, 0, 0]))



        features_opt = []
        features_with_names = {"input_block": [], "middle_block": [], "output_block": []}
        for key in features.keys():
            for feature in features[key]:
                feature = torch.nn.Parameter(feature)
                feature.requires_grad = True
                features_opt.append(feature)
                features_with_names[key].append(feature)
        optimizer = torch.optim.Adam([{'params': features_opt}], lr=fintune_lr)
        pbar = tqdm(range(num_iterations))
        total_loss= 0
        for s in pbar:
            optimizer.zero_grad()
            t, weights = sampler.sample(low_lap.size(0), device=device)
            iterative_loss = diffusion_module.training_losses(model=network, x_start=low_lap, t=t, join_latent=z_sem, model_kwargs={"my_opt": features_with_names})
            mse_loss = torch.mean(iterative_loss['loss'] * weights)
            total_loss = total_loss + mse_loss
            mse_loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss is: {total_loss/(s+1) : .6f}")

        with torch.no_grad():
            _, z_sem = ae_network(F.pad(low_lap, (9,9,9,9,9,9), "constant", 0))
            low_lap = torch.zeros(tuple( [1, 1]+ dwt_sparse_composer.shape_list[config.max_depth])).float().to(device) if low_lap is None else low_lap
            highs_lap = [torch.zeros(tuple( [1, 1]+ dwt_sparse_composer.shape_list[j])).float().to(device) if highs_lap[j] is None else highs_lap[j] for j in range(config.max_depth)]

            voxels_pred = dwt_inverse_3d_lap((low_lap, highs_lap))
            vertices, traingles = mcubes.marching_cubes(voxels_pred.detach().cpu().numpy()[0, 0], 0.0)
            vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
            mcubes.export_off(vertices, traingles, debugger.file_path(f'{testing_sample_index}_gt.off'))

            model_kwargs = {'my_opt':features_with_names}

            low_samples = diffusion_module.ddim_sample_loop(model=network,
                                                            shape=[1, 1] + dwt_sparse_composer.shape_list[-1],
                                                            device=device,
                                                            clip_denoised=clip_noise, progress=True,
                                                            noise=noise,
                                                            eta=ddim_eta,
                                                            model_kwargs=model_kwargs,
                                                            )
            


            highs_samples = [ torch.zeros(tuple([1, 1] + dwt_sparse_composer.shape_list[i]), device = device) for i in range(config.max_depth)]
            if use_detail:
                upsample_low_samples = F.interpolate(low_samples, dwt_sparse_composer.shape_list[2])
                highs_sample = detail_network(upsample_low_samples)
                highs_samples[2] = highs_sample
            

            voxels_pred = dwt_inverse_3d_lap((low_samples, highs_samples))
            vertices, traingles = mcubes.marching_cubes(voxels_pred.detach().cpu().numpy()[0, 0], 0.0)
            vertices = (vertices.astype(np.float32) - 0.5) / config.resolution - 0.5
            mcubes.export_off(vertices, traingles, debugger.file_path(f'{testing_sample_index}_{m%testing_cnt}.off'))




    print("done!")
