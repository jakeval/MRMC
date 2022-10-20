import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

plt.style.use('style.mplstyle')

def load_data(directory):
    if not os.path.exists(f'{directory}/results.csv'):
        df1 = pd.read_pickle(f'{directory}/adult_income.pkl')
        df2 = pd.read_pickle(f'{directory}/german_credit.pkl')
        pd.concat([df1, df2], ignore_index=True).to_csv(f'{directory}/results.csv', index=False)
    df = pd.read_csv(f'{directory}/results.csv')
    return df


def mrmc_compare():
    print("-"*30, "MRMC", "-"*30)
    data = load_data('../mrmc_path_output')
    volcano_mask_1 = (data['alpha_function'] == 'volcano') & (data['alpha_volcano_degree'] == 4)
    all_mask_1 = ((data['k_dirs'] == 4)
               &(~data['immutable_features'].isnull())
               &(data['validate'] == True)
               &(data['early_stopping_cutoff'] == 0.7)
               &(data['sparsity'] == False)
               &(data['step_size'] == 1))
    summary_1 = data[volcano_mask_1 & all_mask_1]

    volcano_mask_2 = (data['alpha_function'] == 'volcano') & (data['alpha_volcano_degree'] == 4)
    all_mask_2 = ((data['k_dirs'] == 4)
               &(~data['immutable_features'].isnull())
               &(data['validate'] == True)
               &(data['early_stopping_cutoff'] == 0.7)
               &(data['sparsity'] == False)
               &(data['step_size'] == 1.25))
    summary_2 = data[volcano_mask_2 & all_mask_2]


    return summary_1, summary_2


def mrmc_visualize(step_size):
    print("-"*30, "MRMC", "-"*30)
    data = load_data('../mrmc_path_output')
    #normal_mask = (data['alpha_function'] == 'normal') & (data['alpha_normal_width'] == 0.5)
    volcano_mask = (data['alpha_function'] == 'volcano') & (data['alpha_volcano_degree'] == 4)
    adult_mask = data['dataset'] == 'adult_income'
    model_mask = data['model'] == 'random_forest'
    all_mask = ((data['k_dirs'] == 4)
               &(~data['immutable_features'].isnull())
               &(data['validate'] == True)
               &(data['early_stopping_cutoff'] == 0.7)
               &(data['sparsity'] == False)
               &(data['step_size'] == step_size))
    summary = data[volcano_mask & all_mask]
    outputs = ['dataset', 'model', 'perturb_dir_random_scale', 'Positive Probability', 'Final Point Distance']
    print(summary[outputs])
    return summary


def dice_visualize(step_size):
    print("-"*30, "DICE", "-"*30)
    data = load_data('../dice_path_output')
    print(data.shape)
    adult_mask = data['dataset'] == 'adult_income'
    model_mask = data['model'] == 'random_forest'
    all_mask = ((~data['immutable_features'].isnull())
               &(data['certainty_cutoff'] == 0.7)
               &(data['perturb_dir_random_scale'] != 16)
               &(data['step_size'] == step_size))
    summary = data[all_mask]
    outputs = ['dataset', 'model', 'perturb_dir_random_scale', 'Positive Probability', 'Final Point Distance']
    print(summary[outputs])
    return summary


def face_visualize(step_size):
    print("-"*30, "FACE", "-"*30)
    data = load_data('../face_path_output')
    print(data.shape)
    adult_mask = (data['dataset'] == 'adult_income') & (data['distance_threshold'] == 2)
    german_mask = (data['dataset'] == 'german_credit') & (data['distance_threshold'] == 8)
    dataset_mask = adult_mask | german_mask
    model_mask = (data['model'] == 'random_forest')
    all_mask = ((~data['immutable_features'].isnull())
               &(data['confidence_threshold'] == 0.7)
               &(data['density_threshold'] == 0)
               &(data['step_size'] == step_size))
    summary = data[all_mask & dataset_mask]
    outputs = ['dataset', 'model', 'perturb_dir_random_scale', 'Positive Probability', 'Final Point Distance']
    #print(summary[outputs])
    #summary.loc[:,'Positive Probability'] = summary.loc[:,'Positive Probability'] * summary.loc[:,'success_ratio']
    print(summary[outputs])
    return summary


def visualize_methods(x, y, dataset, model, df_methods, ax, val_mapper, col_mapper, y_lim=None):
    # df_methods: {method-str: df_output}
    # x-axis: perturb_dir_random_scale
    # y-axis: whatever
    # legend: method
    # title: dataset and model
    dataset_str = val_mapper.get(dataset, dataset)
    model_str = val_mapper.get(model, model)

    for method, df in df_methods.items():
        mask = (df['dataset'] == dataset) & (df['model'] == model)
        df = df[mask]
        x_data = df[x].copy()
        y_data = df[y].copy()
        x_data[x_data.isnull()] = 0
        x_data = x_data.to_numpy()
        y_data = y_data.to_numpy()
        idx = np.argsort(x_data)
        x_data = x_data[idx]
        y_data = y_data[idx]
        ax.plot(x_data, y_data, label=f"{method}")
        x_str = col_mapper.get(x, x)
        y_str = col_mapper.get(y, y)
    ax.set_xlabel(x_str)
    ax.set_ylabel(y_str)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.legend()
    ax.set_title(f"{dataset_str}, {model_str}")


def extreme_example():
    df = pd.read_csv('../mrmc_path_output/poi_stats.csv')
    num_paths = 0
    print(f"POIs with only {num_paths} successful path{'s' if num_paths > 1 else ''}:")
    poi_indices = df[(df['perturb_dir_random_scale'] == 0) & (df['Path Success Count'] == 0)]['poi_index'].unique()
    mask = ((df['perturb_dir_random_scale'] == 0)
            &(df['Path Success Count'] == 0))
    print(df[mask][['poi_index', 'dataset', 'model']])


if __name__ == '__main__':
    print("startup...")
    step_size = 1.5
    dice_df = dice_visualize(step_size)
    compare = False
    if compare:
        df_1, df_2 = mrmc_compare()
        dfs = {
            'd=4': df_1,
            'd=16': df_2,
            'DiCE': dice_df,
        }
        val_mapper = {
            'german_credit': 'German Credit',
            'adult_income': 'Adult Income',
            'random_forest': 'RF',
            'svc': 'SVM'
        }
        col_mapper = {
            'Positive Probability': 'Positive Probability',
            'perturb_dir_random_scale': 'Random Perturbation Standard Deviation'
        }
        settings = [('adult_income', 'random_forest'),
                    ('adult_income', 'svc'),
                    ('german_credit', 'random_forest'),
                    ('german_credit', 'svc')]
        for dataset, model in settings:
            fig, ax = plt.subplots(figsize=(5.5, 3.15))
            visualize_methods(
                'perturb_dir_random_scale',
                'Path Length',
                dataset,
                model,
                dfs,
                ax,
                val_mapper,
                col_mapper,
                #y_lim=(0, 15)
            )
        #plt.show()
        #visualize_table(summary, 'perturb_dir_random_scale', 'Positive Probability', col_mapper, val_mapper, ax, (0,1.1))
        plt.show()
    else:
        mrmc_df = mrmc_visualize(step_size)
        dice_df = dice_visualize(step_size)
        #face_df = face_visualize(step_size)

        dfs = {
            'MRMC': mrmc_df,
            'DiCE': dice_df,
            #'FACE': face_df,
        }
        val_mapper = {
            'german_credit': 'German Credit',
            'adult_income': 'Adult Income',
            'random_forest': 'RF',
            'svc': 'SVM'
        }
        col_mapper = {
            'Positive Probability': 'Positive Probability',
            'perturb_dir_random_scale': 'Random Perturbation Standard Deviation'
        }
        settings = [('adult_income', 'random_forest'),
                    ('adult_income', 'svc'),
                    ('german_credit', 'random_forest'),
                    ('german_credit', 'svc')]
        for dataset, model in settings:
            fig, ax = plt.subplots(figsize=(5.5, 3.15))
            visualize_methods(
                'perturb_dir_random_scale',
                'Final Point Distance',
                dataset,
                model,
                dfs,
                ax,
                val_mapper,
                col_mapper,
                #y_lim=(0, 15)
            )
        #plt.show()
        #visualize_table(summary, 'perturb_dir_random_scale', 'Positive Probability', col_mapper, val_mapper, ax, (0,1.1))
        plt.show()
        # extreme_example()