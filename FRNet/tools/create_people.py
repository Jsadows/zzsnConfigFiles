import argparse
from os import path as osp
from pathlib import Path

import mmengine

total_num = {
    1: 933,
    2: 1010,
    3: 479,
    4: 721,
    5: 426,
    6: 517,
    7: 1121,

}
fold_split = {
    'train': [1, 2, 3, 4],
    'val': [5, 6],
    'trainval': [1, 2, 3, 4, 5, 6],
    'test': [7],
}
split_list = ['train', 'valid', 'trainval', 'test']


def get_semantickitti_info(split: str) -> dict:
    data_infos = dict()
    data_infos['metainfo'] = dict(dataset='SemanticKITTI')
    data_list = []
    for i_folder in fold_split[split]:
        for j in range(total_num[i_folder]):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                    osp.join('sequences',
                             str(i_folder).zfill(2), 'velodyne',
                             str(j).zfill(6) + '.bin'),
                    'num_pts_feats':
                    4
                },
                'pts_semantic_mask_path':
                osp.join('sequences',
                         str(i_folder).zfill(2), 'labels',
                         str(j).zfill(6) + '.label'),
                'sample_idx':
                str(i_folder).zfill(2) + str(j).zfill(6)
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_semantickitti_info_file(pkl_prefix: str, save_path: str) -> None:
    print('Generate info.')
    save_path = Path(save_path)

    semantickitti_infos_train = get_semantickitti_info(split='train')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'SemanticKITTI info train file is saved to {filename}')
    mmengine.dump(semantickitti_infos_train, filename)

    semantickitti_infos_val = get_semantickitti_info(split='val')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'SemanticKITTI info val file is saved to {filename}')
    mmengine.dump(semantickitti_infos_val, filename)

    semantickitti_infos_trainval = get_semantickitti_info(split='trainval')
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'SemanticKITTI info trainval file is saved to {filename}')
    mmengine.dump(semantickitti_infos_trainval, filename)

    semantickitti_infos_test = get_semantickitti_info(split='test')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'SemanticKITTI info test file is saved to {filename}')
    mmengine.dump(semantickitti_infos_test, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/semantickitti',
        required=False,
        help='output path of pkl')
    parser.add_argument('--extra-tag', type=str, default='semantickitti')
    args = parser.parse_args()
    create_semantickitti_info_file(args.extra_tag, args.out_dir)
