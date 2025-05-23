import smplx
import torch
import numpy as np
from . import rotation_conversions as rc
import os
import wget 
import copy

download_path = "./datasets/hub"
smplx_model_dir = os.path.join(download_path, "smplx_models", "smplx")
if not os.path.exists(smplx_model_dir):
    smplx_model_file_path = os.path.join(smplx_model_dir, "SMPLX_NEUTRAL_2020.npz")
    os.makedirs(smplx_model_dir, exist_ok=True)
    if not os.path.exists(smplx_model_file_path):
        print(f"Downloading {smplx_model_file_path}")
        wget.download(
            "https://huggingface.co/spaces/H-Liu1997/EMAGE/resolve/main/EMAGE/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
            smplx_model_file_path,
        )

smplx_model = smplx.create(
    "./datasets/hub/smplx_models/",
    model_type='smplx',
    gender='NEUTRAL_2020',
    use_face_contour=False,
    num_betas=300,
    num_expression_coeffs=100,
    ext='npz',
    use_pca=False,
).eval()

def get_motion_rep_tensor(motion_tensor, pose_fps=30, device="cuda", betas=None):
    global smplx_model
    smplx_model = smplx_model.to(device)
    bs, n, _ = motion_tensor.shape
    motion_tensor = motion_tensor.float().to(device)
    motion_tensor_reshaped = motion_tensor.reshape(bs * n, 165)
    betas = torch.zeros(n, 300, device=device) if betas is None else betas.to(device).unsqueeze(0).repeat(n, 1)
    output = smplx_model(
        betas=torch.zeros(bs * n, 300, device=device),
        transl=torch.zeros(bs * n, 3, device=device),
        expression=torch.zeros(bs * n, 100, device=device),
        jaw_pose=torch.zeros(bs * n, 3, device=device),
        global_orient=torch.zeros(bs * n, 3, device=device),
        body_pose=motion_tensor_reshaped[:, 3:21 * 3 + 3],
        left_hand_pose=motion_tensor_reshaped[:, 25 * 3:40 * 3],
        right_hand_pose=motion_tensor_reshaped[:, 40 * 3:55 * 3],
        return_joints=True,
        leye_pose=torch.zeros(bs * n, 3, device=device),
        reye_pose=torch.zeros(bs * n, 3, device=device),
    )
    joints = output['joints'].reshape(bs, n, 127, 3)[:, :, :55, :]
    dt = 1 / pose_fps
    init_vel = (joints[:, 1:2] - joints[:, 0:1]) / dt
    middle_vel = (joints[:, 2:] - joints[:, :-2]) / (2 * dt)
    final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
    vel = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    position = joints
    rot_matrices = rc.axis_angle_to_matrix(motion_tensor.reshape(bs, n, 55, 3))
    rot6d = rc.matrix_to_rotation_6d(rot_matrices).reshape(bs, n, 55, 6)
    init_vel_ang = (motion_tensor[:, 1:2] - motion_tensor[:, 0:1]) / dt
    middle_vel_ang = (motion_tensor[:, 2:] - motion_tensor[:, :-2]) / (2 * dt)
    final_vel_ang = (motion_tensor[:, -1:] - motion_tensor[:, -2:-1]) / dt
    angular_velocity = torch.cat([init_vel_ang, middle_vel_ang, final_vel_ang], dim=1).reshape(bs, n, 55, 3)
    # position(55*3), vel(55*3), rot6d(55*6), angular_velocity(55*3) => total 55*(3+3+6+3)=55*15
    rep15d = torch.cat([position, vel, rot6d, angular_velocity], dim=3).reshape(bs, n, 55 * 15)
    return {
        "position": position,
        "velocity": vel,
        "rotation": rot6d,
        "axis_angle": motion_tensor,
        "angular_velocity": angular_velocity,
        "rep15d": rep15d,
    }

def get_motion_rep_numpy(poses_np, pose_fps=30, device="cuda", expressions=None, betas=None, trans=None):
    # motion["poses"] is expected to be numpy array of shape (n, 165)
    # (n, 55*3), axis-angle for 55 joints
    global smplx_model
    smplx_model = smplx_model.to(device)
    n = poses_np.shape[0]

    # Convert numpy to torch tensor for SMPL-X forward pass
    poses_ts = torch.from_numpy(poses_np).float().to(device).unsqueeze(0)  # (1, n, 165)
    poses_ts_reshaped = poses_ts.reshape(-1, 165)  # (n, 165)
    betas = torch.zeros(n, 300, device=device) if betas is None else torch.from_numpy(betas).to(device).unsqueeze(0).repeat(n, 1)
    trans = torch.zeros(n, 3, device=device) if trans is None else torch.from_numpy(trans).to(device)
    expressions = torch.zeros(n, 100, device=device) if expressions is None else torch.from_numpy(expressions).float().to(device)

    # Run smplx model to get joints
    output = smplx_model(
        betas=betas,
        transl=trans,
        expression=expressions,
        jaw_pose=poses_ts_reshaped[:, 22 * 3:23 * 3],
        global_orient=poses_ts_reshaped[:, 0:3],
        body_pose=poses_ts_reshaped[:, 3:21 * 3 + 3],
        left_hand_pose=poses_ts_reshaped[:, 25 * 3:40 * 3],
        right_hand_pose=poses_ts_reshaped[:, 40 * 3:55 * 3],
        return_joints=True,
        leye_pose=poses_ts_reshaped[:, 69:72],
        reye_pose=poses_ts_reshaped[:, 72:75],
     )
    joints = output["joints"].detach().cpu().numpy().reshape(n, 127, 3)[:, :55, :]

    betas = betas.cpu().numpy()

    # obtain the trans_vel
    trans = trans.cpu().numpy()
    trans_each_file = trans.copy()
    trans_each_file[:,0] = trans_each_file[:,0] - trans_each_file[0,0]
    trans_each_file[:,2] = trans_each_file[:,2] - trans_each_file[0,2]
    trans_v = np.zeros_like(trans_each_file)
    trans_v[1:,0] = trans_each_file[1:,0] - trans_each_file[:-1,0]
    trans_v[0,0] = trans_v[1,0]
    trans_v[1:,2] = trans_each_file[1:,2] - trans_each_file[:-1,2]
    trans_v[0,2] = trans_v[1,2]
    trans_v[:,1] = trans_each_file[:,1]



    dt = 1 / pose_fps
    # Compute linear velocity
    init_vel = (joints[1:2] - joints[0:1]) / dt
    middle_vel = (joints[2:] - joints[:-2]) / (2 * dt)
    final_vel = (joints[-1:] - joints[-2:-1]) / dt
    vel = np.concatenate([init_vel, middle_vel, final_vel], axis=0)

    position = joints

    # Compute rotation 6D from axis-angle
    poses_ts_reshaped_aa = poses_ts.reshape(1, n, 55, 3)
    rot_matrices = rc.axis_angle_to_matrix(poses_ts_reshaped_aa)[0]  # (n, 55, 3, 3)
    rot6d = rc.matrix_to_rotation_6d(rot_matrices).reshape(n, 55, 6).cpu().numpy()

    # Compute angular velocity
    init_vel_ang = (poses_np[1:2] - poses_np[0:1]) / dt
    middle_vel_ang = (poses_np[2:] - poses_np[:-2]) / (2 * dt)
    final_vel_ang = (poses_np[-1:] - poses_np[-2:-1]) / dt
    angular_velocity = np.concatenate([init_vel_ang, middle_vel_ang, final_vel_ang], axis=0).reshape(n, 55, 3)

    # rep15d: position(55*3), vel(55*3), rot6d(55*6), angular_velocity(55*3) => total 55*(3+3+6+3)=55*15
    rep15d = np.concatenate([position, vel, rot6d, angular_velocity], axis=2).reshape(n, 55 * 15)
    
    return {
        "trans": trans,
        "trans_v": trans_v,
        "position": position,
        "velocity": vel,
        "rotation": rot6d,
        "pose": poses_np,
        "angular_velocity": angular_velocity,
        "shape": betas,
        "rep15d": rep15d,
    }

def process_smplx_motion(pose_file, smplx_model, pose_fps, facial_rep=None):
    """Process SMPLX pose and facial data together."""
    pose_data = np.load(pose_file, allow_pickle=True)
    stride = int(30/pose_fps)
    
    # Extract pose and facial data with same stride
    pose_frames = pose_data["poses"][::stride]
    trans = pose_data["trans"][::stride]
    facial_frames = pose_data["expressions"][::stride] if facial_rep is not None else None
    
    pose_dict = get_motion_rep_numpy(pose_frames, pose_fps=pose_fps, expressions=facial_frames, betas=pose_data["betas"], trans=trans)
    
    if facial_rep is not None:
        pose_dict = {**pose_dict, "facial": facial_frames}
    
    return pose_dict


