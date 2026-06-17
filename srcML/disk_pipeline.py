import pandas as pd
import numpy as np

#cli command:
#smartctl -A -i /dev/sda -j > disk_data.json

def pretvori_json_v_surovi_df(smartctl_dict):

    nujni_stolpci = [
        'smart_1_raw', 'smart_3_raw', 'smart_4_raw', 'smart_5_raw',
        'smart_7_raw', 'smart_9_raw', 'smart_10_raw', 'smart_12_raw',
        'smart_187_raw', 'smart_188_raw', 'smart_190_raw', 'smart_191_raw',
        'smart_192_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw',
        'smart_198_raw', 'smart_199_raw'
    ]

    # Nastavimo privzete NaN vrednosti ( preprocessing bo poskrbel za polnjenje )
    vsebina = {col: np.nan for col in nujni_stolpci}

    vsebina['model'] = smartctl_dict.get('model_name', 'Unknown')
    vsebina['capacity_bytes'] = smartctl_dict.get('user_capacity', {}).get('bytes', 0)
    vsebina['failure'] = 0  # Testna instanca nima odpovedi vnaprej

    # Parsanje ATA SMART tabele iz JSON strukture
    smart_attributes = smartctl_dict.get('ata_smart_attributes', {}).get('table', [])
    for attr in smart_attributes:
        attr_id = attr.get('id')
        col_name = f'smart_{attr_id}_raw'
        if col_name in vsebina:
            vsebina[col_name] = attr.get('raw', {}).get('value', 0)

    return pd.DataFrame([vsebina])


#preprocessing, ki je v "smart_scan_model".. pac podvajanje je.. morm narest reusable metodo
def procesiraj_podatke(df_raw):
    df = df_raw.copy()

    #Pretvorba bajtov v gigabajte
    if 'capacity_bytes' in df.columns:
        df = df.rename(columns={"capacity_bytes": "capacity_gigabytes"})
        df['capacity_gigabytes'] = (df['capacity_gigabytes'] / (1024 ** 3)).round(2)

    #Določanje jeSSD
    if 'jeSSD' not in df.columns and 'model' in df.columns:
        ssd_keywords = ['SSD', 'MTFD', 'SSDSC', '850 PRO', '870 EVO', '860 PRO', '5300']
        df['jeSSD'] = df['model'].apply(lambda x: 1 if any(k in str(x).upper() for k in ssd_keywords) else 0)

    #Zapolnjevanje števcev napak z 0
    stevcne_napake = ['smart_1_raw', 'smart_7_raw', 'smart_192_raw', 'smart_191_raw']
    for col in stevcne_napake:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    #Mehanski cikli (Uporabimo fiksne mediane iz tvoje celotne Backblaze množice)
    mediana_s3, mediana_s4, mediana_s193, mediana_s12 = 0.0, 15.0, 1200.0, 12.0

    if 'smart_3_raw' in df.columns:
        df.loc[(df['jeSSD'] == 1) & (df['smart_3_raw'].isnull()), 'smart_3_raw'] = 0
        df.loc[(df['jeSSD'] == 0) & (df['smart_3_raw'].isnull()), 'smart_3_raw'] = mediana_s3

    if 'smart_4_raw' in df.columns:
        df.loc[(df['jeSSD'] == 1) & (df['smart_4_raw'].isnull()), 'smart_4_raw'] = 0
        df.loc[(df['jeSSD'] == 0) & (df['smart_4_raw'].isnull()), 'smart_4_raw'] = mediana_s4

    if 'smart_193_raw' in df.columns:
        df.loc[(df['jeSSD'] == 1) & (df['smart_193_raw'].isnull()), 'smart_193_raw'] = 0
        df.loc[(df['jeSSD'] == 0) & (df['smart_193_raw'].isnull()), 'smart_193_raw'] = mediana_s193

    if 'smart_12_raw' in df.columns:
        df['smart_12_raw'] = df['smart_12_raw'].fillna(mediana_s12)

    #Zapolnjevanje nujnih ostalih parametrov z 0
    nujni_ostali = ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw']
    for col in nujni_ostali:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    #FEATURE ENGINEERING (any_critical_error, total_error_count, error_per_gb)
    critical_params = ['smart_5_raw', 'smart_187_raw', 'smart_197_raw', 'smart_198_raw']
    df['any_critical_error'] = df[critical_params].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
    df['total_error_count'] = df[critical_params].sum(axis=1)
    df['error_per_gb'] = df['total_error_count'] / (df['capacity_gigabytes'] + 1e-5)

    #Izmet neinformativnih stolpcev
    neinformativni = ['smart_190_raw', 'smart_194_raw', 'smart_199_raw', 'smart_10_raw']
    df.drop(columns=[c for c in neinformativni if c in df.columns], errors='ignore', inplace=True)

    return df

#spet podvajanje iz "smart_scan_model"...
class DiskHealthPipeline:
    def __init__(self, classifier, regressor=None, kmeans=None, scaler=None, expected_columns=None, feature_importance_dict=None):
        self.classifier = classifier
        self.regressor = regressor
        self.kmeans = kmeans
        self.scaler = scaler
        self.expected_columns = expected_columns
        self.feature_importances = feature_importance_dict

    def analyze(self, df_raw):
        #Sprožimo preprocessing
        df_processed = procesiraj_podatke(df_raw)

        #Kategorizacija proizvajalca
        if 'model' in df_processed.columns:
            def getModel(model):
                m = str(model).upper()
                if m.startswith('ST') or 'SEAGATE' in m: return 'Seagate'
                if m.startswith('WDC') or m.startswith('WD') or 'WESTERN' in m: return 'Western Digital'
                if m.startswith('HGST') or m.startswith('HUH') or m.startswith('HMS'): return 'HGST'
                if m.startswith('TOSHIBA') or m.startswith('MG'): return 'Toshiba'
                if m.startswith('SAMSUNG'): return 'Samsung'
                if m.startswith('CT') or 'CRUCIAL' in m: return 'Crucial'
                return 'Other'

            df_processed['model'] = df_processed['model'].apply(getModel)

        #One-hot-encoding poravnava z X_train strukturo
        X_input = pd.DataFrame(0.0, index=[0], columns=self.expected_columns)
        for col in self.expected_columns:
            if col in df_processed.columns:
                X_input.at[0, col] = df_processed[col].iloc[0]
            elif col.startswith('model_'):
                proizvajalec = col.split('model_')[1]
                if 'model' in df_processed.columns and df_processed['model'].iloc[0] == proizvajalec:
                    X_input.at[0, col] = 1

        #Napovedovanje klasifikatorja (Inference)
        K = float(self.classifier.predict(X_input)[0])
        failure_prob = float(self.classifier.predict_proba(X_input)[0][1])

        #HIR FORMULA - failure_probability ze vsebuje vse featere (any_critical_error, starost, ...)
        odstotek_tveganja = round(failure_prob * 100, 2)

        #tveganje ne more biti nikoli 100%, nikoli 0%
        if odstotek_tveganja > 95.0:
            odstotek_tveganja = 97.0
        elif odstotek_tveganja < 5.0:
            odstotek_tveganja = 5.0

        verdict = ""
        if odstotek_tveganja > 75.0:
            verdict = "Critical"
        elif odstotek_tveganja <= 75.0 and odstotek_tveganja > 40.0:
            verdict = "Warning"
        elif odstotek_tveganja <= 40.0:
            verdict = "Healthy"

        return {
            "hir_risk_score": odstotek_tveganja,
            "failure_probability": round(failure_prob, 4),
            "verdict": verdict,
            "models_output": {
                "classification_fail": bool(K == 1.0),
            }
        }