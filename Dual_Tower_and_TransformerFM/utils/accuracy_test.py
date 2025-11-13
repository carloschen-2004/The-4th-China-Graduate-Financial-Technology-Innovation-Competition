import pandas as pd
import os
import ast

# work_dir:
current_script_path = os.path.abspath(__file__)
work_dir = os.path.dirname(os.path.dirname(current_script_path))

# 缓存 merged 表
merge_cache = {}


def get_merged(prod_type):
    if prod_type not in merge_cache:
        df = pd.read_csv(os.path.join(work_dir, f'merged/merged_of_Prod{prod_type}_test.csv'),
                         dtype=str)  # 直接读成字符串省一次转换
        merge_cache[prod_type] = df
    return merge_cache[prod_type]


def compute_accuracy(top_k_custnos, prod_type, prod_id_list):
    df = get_merged(prod_type)

    correct_1 = 0
    correct_2 = 0


    cust_list = top_k_custnos.strip().split('\n')
    num = len(cust_list)

    # 优化：提前筛好成功子表
    df_success = df[df["is_success"] != '0']

    for cust in cust_list:
        cust_dict = ast.literal_eval(cust.strip())  # {'age': '30', 'gender': 'M', ...}

        cols = list(cust_dict.keys())
        values = list(cust_dict.values())

        # 精确匹配（相同列 + 相同值）
        mask_1 = (df[cols] == values).all(axis=1)
        matched = df.loc[mask_1]

        if not matched.empty:
            row = matched.iloc[0]
            # acc_1 条件
            if row["prod_id"] in prod_id_list and row["is_success"] == '1':
                correct_1 += 1

        # 成功的匹配
        mask_2 = (df_success[cols] == values).all(axis=1)
        if mask_2.any():
            correct_2 += 1

    return correct_1 / num, correct_2 / num


if __name__ == '__main__':
    recommend_df = pd.read_excel(os.path.join(work_dir, 'outputs/recommended_user_profiles.xlsx'))

    acc_1_list = []
    acc_2_list = []

    for _, row in recommend_df.iterrows():
        prod_type = row["category"].strip()
        prod_id_list = [x.strip().strip("'").strip() for x in row["prod_id"].strip()[1:-1].split(',')]
        acc_1, acc_2 = compute_accuracy(row["top_k_custnos"], prod_type, prod_id_list)
        acc_1_list.append(acc_1)
        acc_2_list.append(acc_2)

    recommend_df["acc_1"] = acc_1_list
    recommend_df["acc_2"] = acc_2_list
    recommend_df.to_excel(os.path.join(work_dir, 'outputs/recommended_user_profiles_acc.xlsx'), index=False)
    print(f"[Done] saved outputs/recommended_user_profiles_acc.xlsx rows = {len(recommend_df)}")