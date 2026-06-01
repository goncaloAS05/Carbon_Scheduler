from airflow.plugins_manager import AirflowPlugin
import carbon_ex # This imports your listener file

class CarbonMetadataPlugin(AirflowPlugin):
    print("Loading CarbonMetadataPlugin...")
    name = "CarbonMetadataPlugin"
    listeners = [carbon_ex]