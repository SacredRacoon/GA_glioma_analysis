import pandas as pd
import numpy as np
import sklearn as sl


def load_prep(filepath, genes=None):
    df = pd.read_csv(filepath)

    print(f"Размер данных {df.shape}")

    if genes is None:
        genes = ['IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 
                 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR',
                 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA'
                 ]
    for col in genes:
        if col in df.columns:
            df[col] = (df[col]  == 'MUTATED').astype(int)
        else:
            print(f"Не найдена {col}")

    #lgg, gbm это Степени болезни, типо насколько все запущено, глиобластома,
    df['target'] = (df['Grade'] == 'GBM').astype(int)
    lgg_count = (df['target'] == 0).sum()
    gbm_count = (df['target'] == 1).sum()
    print(f"LGG={lgg_count}, GBM={gbm_count}")


    df['age_years'] = df['Age_at_diagnosis'].str.extract(r"(\d+)").astype(float)
    df['sex'] = (df['Gender'] == 'Male').astype(int)

    clean_cols = genes + ['age_years','sex']
    x = df[clean_cols].values
    y = df['target'].values

    print(f"Матрица мутаций {x.shape}")
    top_mutations = df[genes].sum().sort_values(ascending=False).head(5)
    print(f"Топ 5 мутаций {dict(top_mutations)}")

    return x,y,clean_cols,df
if __name__ == "__main__":
    print("Загрузка")
    x, y, cols, df = load_prep("data/raw/TCGA_GBM_LGG_Mutations_all.csv")
    print("Загрузка успешна")