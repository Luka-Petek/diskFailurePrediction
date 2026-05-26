# disk_pipeline.py
import pandas as pd
import numpy as np


class DiskHealthPipeline:
    def __init__(self, classifier, regressor, kmeans, scaler, expected_columns, feature_importance_dict):
        self.classifier = classifier
        self.regressor = regressor
        self.kmeans = kmeans
        self.scaler = scaler
        self.expected_columns = expected_columns
        self.feature_importances = feature_importance_dict

    def analyze(self, df_raw):

        disk_data = df_raw.iloc[0:1].copy()
        X_input = pd.DataFrame(columns=self.expected_columns)

        for col in self.expected_columns:
            if col in disk_data.columns:
                X_input.at[0, col] = disk_data[col].values[0]
            else:
                X_input.at[0, col] = 0.0
        X_input = X_input.fillna(0)

        X_clean = X_input[list(self.scaler.feature_names_in_)]
        X_scaled = self.scaler.transform(X_clean)
        test_clusters = self.kmeans.predict(X_scaled)

        #regresija
        cols_to_drop_reg = ['smart_5_raw', 'model', 'failure', 'total_error_count', 'error_per_gb',
                            'any_critical_error']
        reg_cols = [c for c in self.expected_columns if c not in cols_to_drop_reg]
        X_input_reg = X_input[reg_cols]

        #napovedi
        y_pred = self.classifier.predict(X_input)
        y_pred_r = self.regressor.predict(X_input_reg)

        # Klasifikacija
        K = float(y_pred[0])

        smart5_napoved = y_pred_r[0]
        # Regresija - 50+ sektorjev je največje tveganje (1.0)
        R = min(max(smart5_napoved, 0.0) / 50.0, 1.0)

        cluster_id = test_clusters[0]
        if cluster_id == 1:  # Kriticen
            G = 1.0
        elif cluster_id == 2:  # Ogrozen
            G = 0.5
        else:  # Zdrav
            G = 0.0

        N = float(X_input.iloc[0]['any_critical_error'])

        w_k = 1.5
        w_n = 1.2
        w_g = 0.8
        w_r = 0.5

        vsota_utezi = w_k + w_r + w_g + w_n
        izracun = (w_k * (K ** 2)) + (w_r * (R ** 2)) + (w_g * (G ** 2)) + (w_n * (N ** 2))
        koncni_izracun = np.sqrt(izracun / vsota_utezi)

        odstotek_tveganja = koncni_izracun * 100

        #disk nikol ne more bit 100% ali 0%... slabo narejeno ampak dela
        if odstotek_tveganja > 95.0:
            odstotek_tveganja = 97.0
        elif odstotek_tveganja < 5.0:
            odstotek_tveganja = 5.0

        # Določitev besednega statusa glede na izračunano tveganje
        if odstotek_tveganja > 75.0:
            verdict = "Critical"
        elif odstotek_tveganja > 40.0:
            verdict = "Warning"
        else:
            verdict = "Healthy"

        return {
            "hir_risk_score": round(odstotek_tveganja, 2),
            "verdict": verdict,
            "models_output": {
                "classification_fail": bool(K == 1.0),
                "predicted_smart_5_sectors": round(smart5_napoved, 1),
                "cluster_profile_id": int(cluster_id)
            },
            "critical_features": {
                "smart_5_raw": int(X_input['smart_5_raw'].iloc[0]) if 'smart_5_raw' in X_input.columns else 0,
                "smart_187_raw": int(X_input['smart_187_raw'].iloc[0]) if 'smart_187_raw' in X_input.columns else 0,
                "any_critical_error": int(N)
            },
            "feature_importance": self.feature_importances
        }