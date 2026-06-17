# Backward-compatibility shim — pkl datoteke so bile shranjene ko je bil
# disk_pipeline.py v rootu projekta. Zdaj je implementacija v srcML/sklearn/.
from srcML.sklearn.disk_pipeline import *  # noqa: F401, F403
from srcML.sklearn.disk_pipeline import DiskHealthPipeline, procesiraj_podatke, pretvori_json_v_surovi_df
