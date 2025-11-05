import pandas as pd
import os
import ast

# work_dir: TowTowerNew
current_script_path = os.path.abspath(__file__)
work_dir = os.path.dirname(current_script_path)

recommend_df = pd.read_excel(os.path.join(work_dir, 'outputs/recommended_user_profiles.xlsx'))
acc_1_list = []
acc_2_list = []

for i in range(recommend_df.shape[0]):
    cust_list = recommend_df['top_k_custnos'][i].split('\n')
    prod_type = recommend_df['category'][i].strip()
    prod_id_list = recommend_df['prod_id'][i].strip().strip('[').strip(']').split(',')
    prod_id_list = [s.strip().strip("'").strip() for s in prod_id_list]
    df_merged = pd.read_csv(os.path.join(work_dir,f'merged/merged_of_Prod{prod_type}_test.csv'))
    df_merged = df_merged.astype(str)
    correct_1 = 0
    correct_2 = 0
    num = int(len(cust_list)/2)
    for c in range(num):
        cust = cust_list[c].strip()
        cust_dict = ast.literal_eval(cust)
        cust_df = pd.DataFrame([cust_dict])
        cols = cust_df.columns
        row = cust_df.iloc[0]

        mask_1 = (df_merged[cols] == row.values).all(axis=1)
        matched_row = df_merged.loc[mask_1, "prod_id"]
        prod_id_value = matched_row.iloc[0]
        if prod_id_value in prod_id_list and df_merged.loc[mask_1, "is_success"].iloc[0]=='1':
            correct_1 += 1

        df_merged_success = df_merged[df_merged['is_success'] != '0']
        mask_2 = (df_merged_success[cols] == row.values).all(axis=1)
        if mask_2.any():
            correct_2 += 1
    acc_1 = correct_1/num
    acc_2 = correct_2/num
    acc_1_list.append(acc_1)
    acc_2_list.append(acc_2)
recommend_df['acc_1'] = acc_1_list
recommend_df['acc_2'] = acc_2_list

recommend_df.to_excel(os.path.join(work_dir, 'outputs/recommended_user_profiles_acc.xlsx'))