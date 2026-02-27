import os
import pickle
from tqdm.auto import tqdm
import torch
import numpy as np
from utils.transforms import unNormalize
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
import time
import cv2
import trimesh

def inference(model, infer_dataloader, conf_thresh, results_save_path = None,
                        distributed = False, accelerator = None):
    assert results_save_path is not None
    assert accelerator is not None

    ht,_ = os.path.split(results_save_path)
    results_save_path = ht

    accelerator.print(f'Results will be saved at: {results_save_path}')
    # create output folder if not excist
    os.makedirs(os.path.join(results_save_path,"npz"),exist_ok=True)
    os.makedirs(os.path.join(results_save_path,"overlay"),exist_ok=True)
    #os.makedirs(os.path.join(results_save_path,"image"),exist_ok=True)
    os.makedirs(results_save_path,exist_ok=True)
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model

    progress_bar = tqdm(total=len(infer_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('inference')
    all_time = 0
    for itr, (samples, targets) in enumerate(infer_dataloader):
        start = time.time()
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        print(outputs.keys())
        end = time.time()
        print("Time: "+str(end-start))
        all_time = all_time + (end-start)
        bs = len(targets)
        print(bs)
        for idx in range(bs):
            img_size = targets[idx]['img_size'].detach().cpu().int().numpy()
            img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]

            if(img_name == "000000"):
                camera_params = {'camera_k': np.array([0.0,0.0,0.0,0.0,0.0]).tolist(),'camera_rt': np.array([0.0,0.0,0.0]).tolist(),'camera_c': np.array([outputs['pred_intrinsics'][0][0][0][2].cpu().numpy(),outputs['pred_intrinsics'][0][0][1][2].cpu().numpy()]).tolist(),'camera_f':np.array([outputs['pred_intrinsics'][0][0][0][0].cpu().numpy(),outputs['pred_intrinsics'][0][0][1][1].cpu().numpy()]).tolist(),'height': int(img_size[0]),'width': int(img_size[1]), 'camera_t': np.array([0.0,0.0,0.0]).tolist()}
                with open(os.path.join(results_save_path,"camera.txt"), 'w') as f:
                    f.write(str(camera_params))
                print(camera_params)

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()

            index = select_queries_idx[0].cpu().numpy()
            print(index)
            print(outputs['pred_transl'][idx][index])
            np.savez(os.path.join(results_save_path,"npz", f'{img_name}.npz'),trans=outputs['pred_transl'][idx][index].cpu().numpy(),
                                  root_orient= outputs['pred_poses'][idx][index][0:3].cpu().numpy(),pose_hand = outputs['pred_poses'][idx][index][66:].cpu().numpy(), pose_body = outputs['pred_poses'][idx][index][3:66].cpu().numpy(),
                     betas = np.array([outputs['pred_betas'][idx][index].cpu().numpy()]))


            print(np.array([outputs['pred_betas'][idx][index].cpu().numpy()]))


            ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
            

            #cv2.imwrite(os.path.join(results_save_path,"image", f'{img_name}.jpg'),ori_img)


            ori_img[img_size[0]:,:,:] = 255
            ori_img[:,img_size[1]:,:] = 255
            ori_img[img_size[0]:,img_size[1]:,:] = 255
            ori_img = pad_img(ori_img, model.input_size, pad_color_offset=255)

            sat_img = vis_sat(ori_img.copy(),
                                input_size = model.input_size,
                                patch_size = 14,
                                sat_dict = outputs['sat'],
                                bid = idx)[:img_size[0],:img_size[1]]

            colors = get_colors_rgb(len(pred_verts))
            pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                        verts = pred_verts,
                                        smpl_faces = smpl_layer.faces,
                                        cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                        colors=colors)[:img_size[0],:img_size[1]]
            
            if 'enc_outputs' not in outputs:
                pred_scale_img = np.zeros_like(ori_img)[:img_size[0],:img_size[1]]
            else:
                enc_out = outputs['enc_outputs']
                h, w = enc_out['hw'][idx]
                flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                scale_map = torch.zeros((h,w,2))
                scale_map[ys,xs] = flatten_map

                pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                scale_map = scale_map,
                                                conf_thresh = model.sat_cfg['conf_thresh'],
                                                patch_size=28)[:img_size[0],:img_size[1]]

            pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
            pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
            pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))[:img_size[0],:img_size[1]]

            cv2.imwrite(os.path.join(results_save_path,"overlay", f'{img_name}.png'), pred_mesh_img)


        progress_bar.update(1)
    print("Time: "+str(all_time)) 

