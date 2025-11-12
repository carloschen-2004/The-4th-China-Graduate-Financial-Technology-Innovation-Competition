import pandas as pd
import os
import torch
import ast
from models.TransformerFM_model import ProductTransformerFM

# work_dir: TowTowerNew
current_script_path = os.path.abspath(__file__)
work_dir = os.path.dirname(current_script_path)

first_recommend_df = pd.read_csv(os.path.join(work_dir,'outputs/recommend_custs.csv'))
first_recommend_df.loc[first_recommend_df["rec_cat"] == "T", "rec_cat"] = "C"
unique_cats = first_recommend_df['rec_cat'].unique().tolist()
first_recommend_dict = {}
for cat in unique_cats:
    first_recommend_dict[cat] = first_recommend_df[first_recommend_df['rec_cat'] == cat].copy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型列表
model_files = [f for f in os.path.join(work_dir,"/outputs") if f.startswith('transformer_fm_') and f.endswith('.pth')]
# 所有要输入模型的feature
features = ['event_id', 'event_term', 'event_rate', 'event_amt',
            'is_success_v', 'event_level_A', 'event_level_B', 'event_level_C',
            'prev_count_A', 'prev_count_A_neg', 'prev_count_C', 'prev_count_C_neg',
            'prev_count_D', 'prev_count_D_neg', 'prev_count_N', 'prev_count_N_neg',
            'prev_count_P', 'prev_count_P_neg', 'gender', 'age', 'edu_bg','marriage_situ_cd']
# 给每一个cat的first_recommend_df进行transformer的预测
for cat in unique_cats:
    df = first_recommend_dict[cat]
    df.rename(columns={'prod_id':'prod_id_by_DT'}, inplace=True)
    # 取原test.csv文件
    #path = os.path.join(work_dir,f'merged/merged_of_Prod{cat}_test.csv')
    path = os.path.join(work_dir, f'cleaned/cleaned_event_dataset.csv')
    df_to_merge = pd.read_csv(path)
    # 取出需要合并的列
    cols_to_merge = (['cust_no']+[f'prev_count_{prod}' for prod in unique_cats]+
                     [f'prev_count_{prod}_neg' for prod in unique_cats]+
                     [f'event_level_{level}' for level in ['A','B','C']])
    df_to_merge_clean = df_to_merge[cols_to_merge+['is_success','event_id','prod_id']].copy()
    df_to_merge_clean.rename(columns={'prod_id':'real_prod_id'}, inplace=True)

    # 合并
    df_merged = pd.merge(df, df_to_merge_clean, how='left', on=cols_to_merge)

    # 将所有is_success_v都取1替代is_success输入模型，以免真实结果对其产生影响
    df_merged.loc[:,"is_success_v"] = 1
    # features列表中is_success换成is_success_v
    # features[features.index("is_success")] = "is_success_v"
    # 取出所有feature和保留变量
    cols_kept = ["cust_no","prod_id_by_DT","real_prod_id","is_success"]

    df_merged_to_pred = df_merged[features+cols_kept].copy()

    # 加载模型
    # model = ProductTransformerFM(input_dim=23, hidden_dim=128, num_classes=62)
    model_path = os.path.join(work_dir,"outputs/transformer_fm_product_classifier_{}.pth".format(cat))
    # 1) 读取 checkpoint
    ckpt = torch.load(model_path, map_location=device)
    # 2) 按保存时的参数重新构造模型（非常关键）
    model = ProductTransformerFM(
        input_dim=ckpt['input_dim'],
        hidden_dim=ckpt['hidden_dim'],
        num_classes=ckpt.get('num_items', 1),  # 如果是分类就是类数，如果是回归你存的可能是1
        num_layers=ckpt['num_layers'],
        num_heads=ckpt['num_heads'],
        fm_k=ckpt['fm_k'],
        fusion=ckpt['fusion']
    )
    # 3) 加载训练好的权重
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    print("✅ 模型成功加载完毕！")

    # 转为tensor
    X = df_merged_to_pred[features].astype(float).values
    X = torch.tensor(X, dtype=torch.float32).to(device)


    # 开始分类预测
    with torch.no_grad():
        logits = model(X)  # 输出: [batch, num_classes]
        preds = logits.argmax(dim=1)  # 若分类任务
        preds = preds.cpu().numpy()
        df_merged_to_pred["pred_label"] = preds
        cluster_df = pd.read_excel(os.path.join(work_dir,"cluster_products/first_cluster.xlsx"),sheet_name=cat)
        pred_list = cluster_df["prod_id"]
        df_merged_to_pred["pred_list"] = df_merged_to_pred.apply(
            lambda row: pred_list[row["pred_label"]-2], axis=1
        ).apply(ast.literal_eval)
        # 计算正确率
        df_merged_to_pred['hit'] = df_merged_to_pred.apply(
            lambda row: row["real_prod_id"] in row["pred_list"], axis=1
        ).astype(int)
        acc = (df_merged_to_pred["hit"].sum())/len(df_merged_to_pred)
        print(f"{cat}类二次推荐后的准确率",acc*100,"%")

    os.makedirs(os.path.join(work_dir,"step2_recommend"),exist_ok=True)
    df_merged_to_pred.to_csv(os.path.join(work_dir,f"step2_recommend/recommend_{cat}.csv"))

