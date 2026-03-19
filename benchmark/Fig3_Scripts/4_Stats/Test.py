import pandas as pd
from sklearn.metrics import confusion_matrix

def analyze_infer_sim_13(path):
    # ------------------------------------------------------------------
    # 1. 读取并处理
    # ------------------------------------------------------------------
    df = pd.read_csv(path, sep="\t", header=0)
    df = df[['Infered_Length', 'Score', 'overlap_sim_ratio',
             'Archaic_infered', 'Archaic_sim']]

    # overlap=0 → non-archaic
    df.loc[df['overlap_sim_ratio'] == 0, 'Archaic_sim'] = 0

    # 统一真实标签
    df['Archaic_sim'] = df['Archaic_sim'].replace({'Den': 1, 'Nean': 2}).astype(int)

    # ------------------------------------------------------------------
    # 2. overall archaic ratio
    # ------------------------------------------------------------------
    Archaic_all_count = (df['Archaic_sim'] != 0).mean()

    # ------------------------------------------------------------------
    # 3. 每个推断类别中真实 archaic 比例（hit rate）
    # ------------------------------------------------------------------
    infer_hit_rate = (
        df.groupby('Archaic_infered')['Archaic_sim']
          .apply(lambda x: (x != 0).mean())
    )

    # 若某些 infered 类别不存在，补 0
    infer_hit_rate = infer_hit_rate.reindex([1,2,3], fill_value=0)

    # ------------------------------------------------------------------
    # 4. 混淆矩阵（真实 0/1/2 × 预测 1/2/3）
    # ------------------------------------------------------------------
    y_true = df['Archaic_sim'].astype(int)
    y_pred = df['Archaic_infered'].astype(int)

    labels_pred = [1,2,3]   # columns
    labels_true = [0,1,2]   # rows

    cm = confusion_matrix(y_true, y_pred, labels=labels_pred)

    # cm[i][j] = true = labels_true[i]? pred = labels_pred[j]

    # 展开成 9 个值
    cm_dict = {
        "T0_P1": cm[0][0], "T0_P2": cm[0][1], "T0_P3": cm[0][2],
        "T1_P1": cm[1][0], "T1_P2": cm[1][1], "T1_P3": cm[1][2],
        "T2_P1": cm[2][0], "T2_P2": cm[2][1], "T2_P3": cm[2][2],
    }

    # ------------------------------------------------------------------
    # 5. 返回 13 个值
    # ------------------------------------------------------------------
    return {
        "Archaic_all_count": Archaic_all_count,

        "infer_hit_1": infer_hit_rate.loc[1],
        "infer_hit_2": infer_hit_rate.loc[2],
        "infer_hit_3": infer_hit_rate.loc[3],

        **cm_dict
    }



res = analyze_infer_sim_13("/home/linhuanyu/share1/25_AS_Real/ArchaicSeeker3/500.Simulation_10J19/2_Data/1_Defalut/30133327/infersim.AS3_Merge_0.bed")
print(res)






























