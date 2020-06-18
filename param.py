"""
Borrowed from https://github.com/balakg/posewarp-cvpr2018
"""


def get_general_params():
    param = {}
    dn = 1.0
    param['IMG_HEIGHT'] = int(512/dn)
    param['IMG_WIDTH'] = int(512/dn)
    param['obj_scale_factor'] = 1.14/dn
    param['scale_max'] = 1.05  # Augmentation scaling
    param['scale_min'] = 0.90
    param['max_rotate_degree'] = 5
    param['max_sat_factor'] = 0.05
    param['max_px_shift'] = 10
    param['posemap_downsample'] = 4
    param['sigma_joint'] = 7/4.0
    param['n_joints'] = 14
    param['n_limbs'] = 10

    # Using MPII-style joints: head (0), neck (1), r-shoulder (2), r-elbow (3), r-wrist (4), l-shoulder (5),
    # l-elbow (6), l-wrist (7), r-hip (8), r-knee (9), r-ankle (10), l-hip (11), l-knee (12), l-ankle (13)
    param['limbs'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    param['n_training_iter'] = 200000
    param['test_interval'] = 500
    param['model_save_interval'] = 10
    param['project_dir'] = '/versa/kangliwei/motion_transfer/posewarp-cvpr2018'
    param['model_save_dir'] = param['project_dir'] + '/models'
    # param['data_dir'] = '/versa/kangliwei/motion_transfer/data/posewarp'
    param['data_dir'] = '/versa/kangliwei/motion_transfer/data/masked'
    param['batch_size'] = 8
    return param

if __name__ == '__main__':
    params = get_general_params()
    print(len(params['limbs']))