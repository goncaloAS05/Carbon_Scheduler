import os
import csv
import time
import datetime
import requests


# ==============================
# CONFIGURATION
# ==============================

API_TOKEN = "jdQCGj8jgrv8dswKbgUj"

HEADERS = {
    "auth-token": API_TOKEN
}

CARBON_URL = "https://api.electricitymaps.com/v3/carbon-intensity/past-range"
POWER_URL = "https://api.electricitymaps.com/v3/power-breakdown/past-range"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = BASE_DIR
REGIONS_FILE = os.path.join(BASE_DIR, "regions_123.txt")


# ==============================
# GET ZONE METADATA
# ==============================

print("Fetching zone metadata...")

zone_metadata = {}

try:
    response = requests.get(
        "https://api.electricitymaps.com/v3/zones",
        headers=HEADERS
    )

    if response.status_code == 200:
        zone_metadata = response.json()
    else:
        print("Could not fetch metadata:", response.text)

except Exception as e:
    print("Metadata error:", e)

# ==============================
# LOAD REGIONS
# ==============================

with open(REGIONS_FILE, "r") as f:
    regions = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Found {len(regions)} regions")

# ==============================
# TIME RANGE
# ==============================

start_date = datetime.datetime(
    2025, 1, 1,
    tzinfo=datetime.timezone.utc
)

end_date = datetime.datetime(
    2026, 1, 1,
    tzinfo=datetime.timezone.utc
)

# ==============================
# CSV FORMAT
# ==============================

CSV_HEADERS = [
    "Datetime (UTC)",
    "Country",
    "Zone name",
    "Zone id",
    "Carbon intensity gCO₂eq/kWh (direct)",
    "Carbon intensity gCO₂eq/kWh (Life cycle)",
    "Carbon-free energy percentage (CFE%)",
    "Renewable energy percentage (RE%)",
    "Data source",
    "Data estimated",
    "Data estimation method"
]

# ==============================
# HELPERS
# ==============================


def extract_carbon_metrics(carbon, power):
    direct_ci = carbon.get("carbonIntensity", "")

    lifecycle_ci = (
        carbon.get("carbonIntensityLca")
        or carbon.get("carbonIntensity")
        or carbon.get("lifecycleCarbonIntensity")
        or ""
    )

    cfe = power.get("carbonFreePercentage", "")
    renewable = power.get("renewablePercentage", "")
    source = carbon.get("source", "")
    estimated = carbon.get("isEstimated", False)
    estimation_method = carbon.get("estimationMethod") or ""

    return {
        "direct_ci": direct_ci,
        "lifecycle_ci": lifecycle_ci,
        "cfe": cfe,
        "renewable": renewable,
        "source": source,
        "estimated": estimated,
        "estimation_method": estimation_method,
    }


# ==============================
# DOWNLOAD LOOP
# ==============================

for zone in regions:

    filename = os.path.join(
        OUTPUT_DIR,
        f"log_{zone}.csv"
    )

    print("\nDownloading", zone)

    zone_info = zone_metadata.get(zone, {})

    country = zone_info.get(
        "countryName",
        zone
    )

    zone_name = zone_info.get(
        "zoneName",
        zone
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADERS)

        current_start = start_date

        while current_start < end_date:

            current_end = min(
                current_start + datetime.timedelta(days=10),
                end_date
            )

            params = {
                "zone": zone,
                "start": current_start.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "end": current_end.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "temporalGranularity": "hourly"
            }

            try:

                # --------------------------
                # Carbon intensity requests
                # --------------------------

                direct_params = dict(params)
                direct_params["emissionFactorType"] = "direct"

                lifecycle_params = dict(params)
                lifecycle_params["emissionFactorType"] = "lifecycle"

                carbon_direct_response = requests.get(
                    CARBON_URL,
                    headers=HEADERS,
                    params=direct_params
                )

                carbon_lifecycle_response = requests.get(
                    CARBON_URL,
                    headers=HEADERS,
                    params=lifecycle_params
                )

                # --------------------------
                # Power breakdown request
                # --------------------------

                power_response = requests.get(
                    POWER_URL,
                    headers=HEADERS,
                    params=params
                )

                if carbon_direct_response.status_code != 200 or carbon_lifecycle_response.status_code != 200:

                    print(
                        " Carbon error:",
                        "direct=",
                        carbon_direct_response.status_code,
                        "lifecycle=",
                        carbon_lifecycle_response.status_code,
                        carbon_direct_response.text,
                        carbon_lifecycle_response.text
                    )

                    current_start = current_end
                    continue

                carbon_direct_json = carbon_direct_response.json()
                carbon_lifecycle_json = carbon_lifecycle_response.json()
                power_json = (
                    power_response.json()
                    if power_response.status_code == 200
                    else {}
                )

                # Convert lists into dictionaries by datetime

                direct_carbon_data = {
                    x["datetime"]: x
                    for x in carbon_direct_json.get(
                        "data",
                        []
                    )
                }

                lifecycle_carbon_data = {
                    x["datetime"]: x
                    for x in carbon_lifecycle_json.get(
                        "data",
                        []
                    )
                }

                power_data = {
                    x["datetime"]: x
                    for x in power_json.get(
                        "data",
                        []
                    )
                }

                # --------------------------
                # Merge datasets
                # --------------------------

                for dt in sorted(set(direct_carbon_data) | set(lifecycle_carbon_data)):

                    direct_carbon = direct_carbon_data.get(dt, {})
                    lifecycle_carbon = lifecycle_carbon_data.get(dt, {})
                    power = power_data.get(
                        dt,
                        {}
                    )

                    direct_metrics = extract_carbon_metrics(
                        direct_carbon,
                        power
                    )
                    lifecycle_metrics = extract_carbon_metrics(
                        lifecycle_carbon,
                        power
                    )

                    direct_ci = direct_metrics["direct_ci"]
                    lifecycle_ci = lifecycle_metrics["lifecycle_ci"]
                    cfe = direct_metrics["cfe"]
                    renewable = direct_metrics["renewable"]
                    source = lifecycle_metrics["source"]
                    estimated = lifecycle_metrics["estimated"]
                    estimation_method = lifecycle_metrics["estimation_method"]


                    writer.writerow([
                        dt,
                        country,
                        zone_name,
                        zone,
                        float(direct_ci) if direct_ci != "" else "",
                        float(lifecycle_ci) if lifecycle_ci != "" else "",
                        float(cfe) if cfe != "" else "",
                        float(renewable) if renewable != "" else "",
                        source,
                        estimated,
                        estimation_method
                    ])
            except Exception as e:
                print(
                    " Exception:",
                    zone,
                    e
                )
            current_start = current_end
            # avoid API throttling
            time.sleep(0.5)

    print(
        "Saved:",
        filename
    )


print("\nFinished! All CSV files are in:")
print(OUTPUT_DIR)