import math
from data.names import names
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, normalize


ONE_FRAME_LENGTH = 0.0333


def calculate_eccentricity(df):
    results = []

    for group, data in df.groupby('j'):
        points = data[['x', 'y', 'z']].to_numpy()

        mean = np.mean(points, axis=0)
        centered_points = points - mean

        if len(centered_points) > 1:
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, _ = np.linalg.eig(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]

            if eigenvalues[0] > 0:
                eccentricity = np.sqrt(1 - (eigenvalues[1] / eigenvalues[0]))
            else:
                eccentricity = 1.0

        else:
            eccentricity = 1.0
        results.append({'group': group, 'eccentricity': eccentricity.real})
    eccentricity_df = pd.DataFrame(results)

    df['ecc'] = df['j'].map(eccentricity_df.set_index('group')['eccentricity'])

    return df


mapping = {0: 'flash_length',
           1: 'flash_gap',
           2: 'flash_count',
           3: 'x',
           4: 'y',
           5: 'z',
           6: 'avx',
           7: 'avy',
           8: 'avz',
           9: 'v',
           10: 'totm',
           11: 'stk',
           12: 'traj',
           13: 'timeseries',
           14: 'folder_name',
           15: 'date_str',
           16: 'species_label',
           17: 'species',
           18: 'ecc'}

r_map = {
    'flash_length': 0,
    'flash_gap': 1,
    'flash_count': 2,
    'x': 3,
    'y': 4,
    'z': 5,
    'avx': 6,
    'avy': 7,
    'avz': 8,
    'v': 9,
    'totm': 10,
    'stk': 11,
    'traj': 12,
    'timeseries': 13,
    'folder_name': 14,
    'date_str': 15,
    'species_label': 16,
    'species': 17,
    'ecc': 18
}


def find_closest_time(times, target_time):
    idx = np.abs(times - target_time).argmin()
    return times[idx]


def get_label_species_from_filename(f):
    if 'ic' in f:
        return names.name_label_dict[names.short_long_dict['ic']], names.short_long_dict['ic']
    elif 'bw' in f:
        return names.name_label_dict[names.short_long_dict['bw']], names.short_long_dict['bw']
    elif 'xx' in f:
        return names.name_label_dict[names.short_long_dict['xx']], names.short_long_dict['xx']
    elif 'mx' in f:
        return names.name_label_dict[names.short_long_dict['mx']], names.short_long_dict['mx']
    elif 'ldc' in f:
        return names.name_label_dict[names.short_long_dict['us']], names.short_long_dict['us']
    elif 'rbp' in f:
        return names.name_label_dict[names.short_long_dict['us']], names.short_long_dict['us']
    elif 'saw' in f:
        return names.name_label_dict[names.short_long_dict['us']], names.short_long_dict['us']
    elif 'ub' in f:
        return names.name_label_dict[names.short_long_dict['ub']], names.short_long_dict['ub']
    elif 'ur' in f:
        return names.name_label_dict[names.short_long_dict['ur']], names.short_long_dict['ur']
    elif 'io' in f:
        return names.name_label_dict[names.short_long_dict['io']], names.short_long_dict['io']
    elif 'uf' in f:
        return names.name_label_dict[names.short_long_dict['uf']], names.short_long_dict['uf']
    elif 'ik' in f:
        return names.name_label_dict[names.short_long_dict['ik']], names.short_long_dict['ik']
    elif 'uw' in f:
        return names.name_label_dict[names.short_long_dict['uw']], names.short_long_dict['uw']
    else:
        return 'Not in dataset yet! ', 'None'


def calc_displacements(x, y, z):
    displacements = []
    first_point = (x.iloc[0], y.iloc[0], z.iloc[0])
    for i in enumerate(zip(x, y, z)):
        if i[0] == 0:
            first_point = i[1]
        else:
            last_point = i[1]
            displacements.append(
                math.sqrt((last_point[0] - first_point[0])**2 +  # x2 - x1 ^ 2
                          (last_point[1] - first_point[1])**2 +  # y2 - y1 ^ 2
                          (last_point[2] - first_point[2])**2)   # z2 - z1 ^ 2
            )
            first_point = i[1]

    return displacements


def trim_and_collate(cdf, dfc):

    numeric_headers = ['flash_length', 'flash_gap', 'flash_count', 'v', 'totm', 'ecc']
    headers = list(r_map.keys())

    cdf = pd.DataFrame(cdf, columns=headers)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform([
        tuple(x[h] for h in numeric_headers)
        for x in cdf
    ])
    shifted_data = scaled_data + abs(np.min(scaled_data)) + 1  # Make positive
    log_transformed_data = np.log(shifted_data)

    normed_data = normalize(log_transformed_data, norm='l2')
    normed_data = pd.DataFrame(normed_data, columns=headers)

    # cluster logic follows:
    # this will be where we restrict / relabel aspects of the data depending on cluster membership... but only once that
    # is all up and running

    # print('Mapping trajectories to clusters...\r')
    # mapping_dict = cdf.set_index('traj')['cluster'].to_dict()
    # dfc['cluster'] = dfc['j'].map(mapping_dict)
    #
    # value_counts = dfc['cluster'].value_counts()
    #
    # # keep only clusters that are more prevalent than 1% of the data
    # values_to_keep = value_counts[value_counts >= (len(dfc) / 100)].index.tolist()
    # result_df = dfc[dfc['cluster'].isin(values_to_keep)]
    # print('...done')
    #
    # # drop the noise cluster
    # feature_df = cdf[cdf['cluster'].isin(values_to_keep)]
    # result_df = result_df.loc[result_df['cluster'] != -1]
    # feature_df = feature_df.loc[feature_df['cluster'] != -1]
    # cluster_splits = list(set(result_df.cluster.values))
    # print('Filtering out clusters that occur < 1% of the activity...\r')
    #
    # # Calculate instance percentage, and filter by occurrence percent
    # instance_percentages = {
    #     cluster: round(len(result_df[result_df['cluster'] == cluster]) / len(result_df), 3) * 100
    #     for cluster in cluster_splits
    # }
    # filtered_clusters = {
    #     cluster: percent for cluster, percent in instance_percentages.items() if percent >= 1.0
    # }
    # filtered_clusters = list(filtered_clusters.keys())
    #
    # feature_df = feature_df[feature_df['cluster'].isin(filtered_clusters)]
    # result_df = result_df[result_df['cluster'].isin(filtered_clusters)]
    cdf[numeric_headers] = normed_data[numeric_headers].values

    feature_df = cdf
    result_df = dfc

    datemapping = cdf.set_index('traj')['date_str'].to_dict()
    feature_df['Dataset'] = feature_df['traj'].map(datemapping)
    print('...done')
    return feature_df, result_df


def extract_from_csv(f, file_flag=True):
    if file_flag:
        if '.csv' in f:
            df = pd.read_csv(f, names=['x', 'y', 'z', 't', 'k', 'j'])
            label, species = get_label_species_from_filename(f)
            df_cleaned = df[df.z > -1.0]
            # remove potentially distorted points near the x-y origin
            df_cleaned = df_cleaned[(np.sqrt(df_cleaned.x ** 2 + df_cleaned.y ** 2) > 0.4)]
            grouped_by_traj_df = df_cleaned.groupby('j')
            singleflashlengths = []
            singleflashpos = []
            singlecombined = []
            savename = f.split('/')[-1].split('.')[0]

            flashlengths = []
            flashpos = []
            combined = []
            for group_name, group_data in grouped_by_traj_df:
                #  single flash in the trajectory
                if len(list(set(group_data.k.values))) == 1:
                    if max(group_data.t) != min(group_data.t):
                        traj_times = np.sort(group_data.t.values)
                        min_t, max_t = traj_times.min(), traj_times.max()
                        time_grid = np.arange(min_t, max_t + 1 / 30, 1 / 30)
                        closest_times = np.array([find_closest_time(traj_times, t) for t in time_grid])
                        time_diff = np.abs(time_grid - closest_times)
                        timeseries = (time_diff <= 1 / 30).astype(int)
                        singleflashlengths.append(max(group_data.t) - min(group_data.t))
                        singleflashpos.append((np.mean(group_data.x), np.mean(group_data.y), np.mean(group_data.z)))
                        flashlengths.append(max(group_data.t) - min(group_data.t))
                        flashpos.append((np.mean(group_data.x), np.mean(group_data.y), np.mean(group_data.z)))
                        singlecombined.append((max(group_data.t) - min(group_data.t), np.mean(group_data.x),
                                               np.mean(group_data.y), np.mean(group_data.z)))
                        combined.append((max(group_data.t) - min(group_data.t),  # flash length [0]
                                         0,  # flash gap [1]
                                         1,  # num flashes [2]
                                         [np.mean(group_data.x)],  # x [6]
                                         [np.mean(group_data.y)],  # y [7]
                                         [np.mean(group_data.z)],  # z [8],
                                         np.mean(group_data.x),  # avx [6]
                                         np.mean(group_data.y),  # avy [7]
                                         np.mean(group_data.z),  # avz [8],

                                         np.mean(calc_displacements(group_data.x, group_data.y, group_data.z)),  # v [9],
                                         np.sum(calc_displacements(group_data.x, group_data.y, group_data.z)),  # sum_v[10]
                                         group_data.k.values[0],  # k [11],
                                         group_name,  # j [12]
                                         timeseries,  # timeseries [13]
                                         f.split('/')[-1],  # folder_name [14]
                                         savename,  # date [15]
                                         label,  # label [16]
                                         species # species [17]
                                         ))

                else:
                    trajectory_v = calc_displacements(group_data.x, group_data.y, group_data.z)
                    m_displacement = np.mean(trajectory_v)
                    t_displacement = np.sum(trajectory_v)
                    flash_count = len(list(set(group_data.k.values)))
                    flash_gaps = []
                    flash_lengths = []
                    ks = []
                    xs = []
                    ys = []
                    zs = []
                    first_group = None
                    for kgroup_name, kgroup_data in group_data.groupby('k'):
                        if first_group is None:
                            first_group = kgroup_data

                        if max(first_group.t) != max(kgroup_data.t):
                            flash_gap = min(kgroup_data.t) - max(first_group.t)
                            if flash_gap > 2*ONE_FRAME_LENGTH:
                                ks.append(kgroup_name)
                                flash_gaps.append(flash_gap)

                                flash_lengths.append(max(kgroup_data.t) - min(kgroup_data.t))
                                flashlengths.append(max(kgroup_data.t) - min(kgroup_data.t))
                                xs.append(kgroup_data.x)
                                ys.append(kgroup_data.y)
                                zs.append(kgroup_data.z)

                                flashpos.append((np.mean(kgroup_data.x), np.mean(kgroup_data.y), np.mean(kgroup_data.z)))
                                first_group = kgroup_data
                            else:
                                ks.extend(list(set(kgroup_data.k.values)))
                                # Add current 'k' to ks, indicating merged group
                                group_data.loc[kgroup_data.index, 'k'] = first_group['k'].values[0]

                        else:
                            flash_lengths.append(max(kgroup_data.t) - min(kgroup_data.t))
                            flashlengths.append(max(kgroup_data.t) - min(kgroup_data.t))
                            xs.append(kgroup_data.x)
                            ys.append(kgroup_data.y)
                            zs.append(kgroup_data.z)

                            flashpos.append((np.mean(kgroup_data.x), np.mean(kgroup_data.y), np.mean(kgroup_data.z)))
                            first_group = kgroup_data

                    flash_gap = np.max(flash_gaps) if len(flash_gaps) > 0 else 0

                    if len(flash_lengths) == 0:
                        flash_length = 0
                    else:
                        flash_length = np.max(flash_lengths)

                    if flash_length > ONE_FRAME_LENGTH:
                        traj_times = np.sort(group_data.t.values)
                        min_t, max_t = traj_times.min(), traj_times.max()
                        time_grid = np.arange(min_t, max_t + 1 / 30, 1 / 30)
                        closest_times = np.array([find_closest_time(traj_times, t) for t in time_grid])
                        time_diff = np.abs(time_grid - closest_times)
                        timeseries = (time_diff <= 1 / 30).astype(int)
                        combined.append((flash_length,  # flash length [0]
                                         flash_gap,  # flash gap [1] will never be less than 0 so this is a good sentinel
                                         flash_count,  # flash count [2]
                                         xs,  # x [3]
                                         ys,  # y [4]
                                         zs,  # z [5]
                                         np.mean(group_data.x),  # avx [6]
                                         np.mean(group_data.y),  # avy [7]
                                         np.mean(group_data.z),  # avz [8]
                                         m_displacement,   # v [9],
                                         t_displacement,   # sumv[10]
                                         list(set(ks)),   # k [11]
                                         group_name,     # j [12]
                                         timeseries,     # timeseries [13]
                                         f.split('/')[-1],  # folder_name [14]
                                         savename,   # date [15]
                                         label,  # label [16]
                                         species))    # species [17]
            return df_cleaned, combined

    else:
        # This lets you pass it a 'csvs' folder like the other stuff
        all_dfs = []
        all_combined = []
        from pathlib import Path
        folders = [Path(f) for f in names.dataset_folders]
        for folder in folders:
            for file in folder.iterdir():
                if '.csv' in file:
                    df = pd.read_csv(file, names=['x', 'y', 'z', 't', 'k', 'j'])
                    label, species = get_label_species_from_filename(str(file))
                    df_cleaned = df[df.z > -1.0]
                    # remove potentially distorted points near the x-y origin
                    df_cleaned = df_cleaned[(np.sqrt(df_cleaned.x ** 2 + df_cleaned.y ** 2) > 0.4)]
                    grouped_by_traj_df = df_cleaned.groupby('j')
                    singleflashlengths = []
                    singleflashpos = []
                    singlecombined = []
                    savename = str(file).split('/')[-1].split('.')[0]

                    flashlengths = []
                    flashpos = []
                    combined = []
                    for group_name, group_data in grouped_by_traj_df:
                        #  single flash in the trajectory
                        if len(list(set(group_data.k.values))) == 1:
                            if max(group_data.t) != min(group_data.t):
                                traj_times = np.sort(group_data.t.values)
                                min_t, max_t = traj_times.min(), traj_times.max()
                                time_grid = np.arange(min_t, max_t + 1 / 30, 1 / 30)
                                closest_times = np.array([find_closest_time(traj_times, t) for t in time_grid])
                                time_diff = np.abs(time_grid - closest_times)
                                timeseries = (time_diff <= 1 / 30).astype(int)
                                singleflashlengths.append(max(group_data.t) - min(group_data.t))
                                singleflashpos.append((np.mean(group_data.x), np.mean(group_data.y), np.mean(group_data.z)))
                                flashlengths.append(max(group_data.t) - min(group_data.t))
                                flashpos.append((np.mean(group_data.x), np.mean(group_data.y), np.mean(group_data.z)))
                                singlecombined.append((max(group_data.t) - min(group_data.t), np.mean(group_data.x),
                                                       np.mean(group_data.y), np.mean(group_data.z)))
                                combined.append((max(group_data.t) - min(group_data.t),  # flash length [0]
                                                 0,  # flash gap [1]
                                                 1,  # num flashes [2]
                                                 [np.mean(group_data.x)],  # x [6]
                                                 [np.mean(group_data.y)],  # y [7]
                                                 [np.mean(group_data.z)],  # z [8],
                                                 np.mean(group_data.x),  # avx [6]
                                                 np.mean(group_data.y),  # avy [7]
                                                 np.mean(group_data.z),  # avz [8],

                                                 np.mean(calc_displacements(group_data.x, group_data.y, group_data.z)),
                                                 # v [9],
                                                 np.sum(calc_displacements(group_data.x, group_data.y, group_data.z)),
                                                 # sum_v[10]
                                                 group_data.k.values[0],  # k [11],
                                                 group_name,  # j [12]
                                                 timeseries,  # timeseries [13]
                                                 str(file).split('/')[-1],  # folder_name [14]
                                                 savename,  # date [15]
                                                 label,  # label [16]
                                                 species))  # species [17]

                        else:
                            trajectory_v = calc_displacements(group_data.x, group_data.y, group_data.z)
                            m_displacement = np.mean(trajectory_v)
                            t_displacement = np.sum(trajectory_v)
                            flash_count = len(list(set(group_data.k.values)))
                            flash_gaps = []
                            flash_lengths = []
                            ks = []
                            xs = []
                            ys = []
                            zs = []
                            first_group = None
                            for kgroup_name, kgroup_data in group_data.groupby('k'):
                                if first_group is None:
                                    first_group = kgroup_data

                                if max(first_group.t) != max(kgroup_data.t):
                                    flash_gap = min(kgroup_data.t) - max(first_group.t)
                                    if flash_gap > 2 * ONE_FRAME_LENGTH:
                                        ks.append(kgroup_name)
                                        flash_gaps.append(flash_gap)

                                        flash_lengths.append(max(kgroup_data.t) - min(kgroup_data.t))
                                        flashlengths.append(max(kgroup_data.t) - min(kgroup_data.t))
                                        xs.append(kgroup_data.x)
                                        ys.append(kgroup_data.y)
                                        zs.append(kgroup_data.z)

                                        flashpos.append(
                                            (np.mean(kgroup_data.x), np.mean(kgroup_data.y), np.mean(kgroup_data.z)))
                                        first_group = kgroup_data
                                    else:
                                        ks.extend(list(set(kgroup_data.k.values)))
                                        # Add current 'k' to ks, indicating merged group
                                        group_data.loc[kgroup_data.index, 'k'] = first_group['k'].values[0]

                                else:
                                    flash_lengths.append(max(kgroup_data.t) - min(kgroup_data.t))
                                    flashlengths.append(max(kgroup_data.t) - min(kgroup_data.t))
                                    xs.append(kgroup_data.x)
                                    ys.append(kgroup_data.y)
                                    zs.append(kgroup_data.z)

                                    flashpos.append(
                                        (np.mean(kgroup_data.x), np.mean(kgroup_data.y), np.mean(kgroup_data.z)))
                                    first_group = kgroup_data

                            flash_gap = np.max(flash_gaps) if len(flash_gaps) > 0 else 0

                            if len(flash_lengths) == 0:
                                flash_length = 0
                            else:
                                flash_length = np.max(flash_lengths)

                            if flash_length > ONE_FRAME_LENGTH:
                                traj_times = np.sort(group_data.t.values)
                                min_t, max_t = traj_times.min(), traj_times.max()
                                time_grid = np.arange(min_t, max_t + 1 / 30, 1 / 30)
                                closest_times = np.array([find_closest_time(traj_times, t) for t in time_grid])
                                time_diff = np.abs(time_grid - closest_times)
                                timeseries = (time_diff <= 1 / 30).astype(int)
                                combined.append((flash_length,  # flash length [0]
                                                 flash_gap,
                                                 # flash gap [1] will never be less than 0 so this is a good sentinel
                                                 flash_count,  # flash count [2]
                                                 xs,  # x [3]
                                                 ys,  # y [4]
                                                 zs,  # z [5]
                                                 np.mean(group_data.x),  # avx [6]
                                                 np.mean(group_data.y),  # avy [7]
                                                 np.mean(group_data.z),  # avz [8]
                                                 m_displacement,  # v [9],
                                                 t_displacement,  # sumv[10]
                                                 list(set(ks)),  # k [11]
                                                 group_name,  # j [12]
                                                 timeseries,  # timeseries [13]
                                                 str(file).split('/')[-1],  # folder_name [14]
                                                 savename,  # date [15]
                                                 label,  # label [16]
                                                 species))  # species [17]

                    all_dfs.append(df_cleaned)
                    all_combined.extend(combined)
        final_df = pd.concat(all_dfs, ignore_index=True)
        return final_df, all_combined

