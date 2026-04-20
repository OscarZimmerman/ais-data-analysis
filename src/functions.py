import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.neighbors import BallTree
import os
import folium



def percentile_rank(series):
    return series.rank(pct=True)


def ais_gap_analysis(df):
    df = df.copy()
    df['time_delta'] = df.groupby('MMSI')['BaseDateTime'].diff().dt.total_seconds()/60  # Time delta in minutes

    df['minor_gap_flag'] = (df['time_delta'] > 10 ) & (df['time_delta'] <= 30)
    df['major_gap_flag'] = (df['time_delta'] > 30 ) & (df['time_delta'] <= 180)
    df['dark_gap_flag'] = (df['time_delta'] > 180 ) 

    gap_summary = df.groupby('MMSI').agg(
        minor_gaps=('minor_gap_flag','sum'),
        major_gaps=('major_gap_flag','sum'),
        dark_gaps=('dark_gap_flag','sum'),
        avg_gap=('time_delta','mean')
    )
    

    # Weighted gap count: minor=1, major=2, dark=4 
    gap_summary['gap_count'] = (
        gap_summary['minor_gaps'] * 1 +
        gap_summary['major_gaps'] * 2 +
        gap_summary['dark_gaps'] * 4
    )

    return gap_summary




def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))




def detect_sts_events(df, distance_km=0.3):
    print("Starting STS detection")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["BaseDateTime"])

    print("Total AIS points:", len(df))
    print("Unique vessels:", df["MMSI"].nunique())

    print("\nFiltering low-speed AIS points (SOG < 2 knots)")
    sts_candidates = df[df["SOG"] < 2].copy()

    print("Candidate AIS points:", len(sts_candidates))


    sts_candidates["time_bin"] = sts_candidates["timestamp"].dt.floor("30min") # Time binning (30 minutes)


    GRID_SIZE = 0.05 # ~5km grid cells at the equator
    sts_candidates["lat_bin"] = (sts_candidates["LAT"] / GRID_SIZE).astype(int)
    sts_candidates["lon_bin"] = (sts_candidates["LON"] / GRID_SIZE).astype(int)


    groups = sts_candidates.groupby(["time_bin", "lat_bin", "lon_bin"]) # Group by time and spatial bins

    print("Total space-time groups:", len(groups))

    events = [] # List to store candidate STS events

    for i, ((time_bin, lat_bin, lon_bin), subset) in enumerate(groups):

        if i % 500 == 0:
            print(f"Processing group {i}/{len(groups)}")

        if subset["MMSI"].nunique() < 2: # Need at least 2 vessels to have an encounter
            continue

        coords = subset[["LAT", "LON"]].values 
        mmsi = subset["MMSI"].values

        n = len(subset)

        for a in range(n):
            for b in range(a + 1, n):

                if mmsi[a] == mmsi[b]: # Skip same vessel
                    continue

                dist = haversine(            # Calculate distance between the two points
                    coords[a][0], coords[a][1],
                    coords[b][0], coords[b][1]
                )

                if dist < distance_km: # If within distance threshold, record as candidate STS event
                    events.append({
                        "MMSI1": mmsi[a],
                        "MMSI2": mmsi[b],
                        "time_bin": time_bin
                    })

    print("\nSTS scanning complete")
    print("Total candidate encounters:", len(events))

    sts_df = pd.DataFrame(events)

    if not sts_df.empty: # Count STS events per vessel
        sts_counts_1 = sts_df.groupby("MMSI1").size() 
        sts_counts_2 = sts_df.groupby("MMSI2").size()
        sts_counts = sts_counts_1.add(sts_counts_2, fill_value=0) # Sum counts from both sides of the encounter
        sts_counts = sts_counts.rename("STS_Count")
        sts_counts = sts_counts.reindex(df["MMSI"].unique()).fillna(0)
    else:
        sts_counts = pd.Series(0, index=df["MMSI"].unique(), name="STS_Count")

    print("STS vessels detected:", (sts_counts > 0).sum())
    print("STS detection finished")

    return sts_df, sts_counts





def route_irregularity_analysis(df):
   
    df = df.copy()


    df["lat_prev"] = df.groupby("MMSI")["LAT"].shift()
    df["lon_prev"] = df.groupby("MMSI")["LON"].shift()


    # Distance between consecutive points

    df["dist_km"] = haversine(
        df["LAT"],
        df["LON"],
        df["lat_prev"],
        df["lon_prev"]
    )

    
    # Route irregularity = std of distances
   
    route_irregularity = (
        df.groupby("MMSI")["dist_km"]
        .std()
        .fillna(0)
        .rename("route_irregularity")
    )

    return route_irregularity





def name_change_analysis(df):
    

    df = df.copy() # Avoid modifying original DataFrame
 

    df["VesselName"] = df["VesselName"].astype(str)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

    # Count unique names per MMSI
  
    name_counts = df.groupby("MMSI")["VesselName"].nunique()
    multi_name_vessels = name_counts[name_counts > 1]


    # Detect name change events

    df = df.sort_values(["MMSI", "BaseDateTime"])

    df["previous_name"] = df.groupby("MMSI")["VesselName"].shift()

    df["name_change_flag"] = df["VesselName"] != df["previous_name"]

    # Remove first occurrence (NaN comparison issue)
    df.loc[df["previous_name"].isna(), "name_change_flag"] = False

    name_change_events = df[df["name_change_flag"]][
        ["MMSI", "BaseDateTime", "previous_name", "VesselName"]
    ]


    # Count changes per vessel

    name_change_counts = (
        name_change_events.groupby("MMSI")
        .size()
        .rename("Name_Change_Count")
    )

    return name_change_events, name_change_counts, multi_name_vessels


def compute_vessel_risk(df, gap_summary, weights=None):

    if weights is None:
        weights = {
            "gap": 0.35,
            "route": 0.2,
            "sts": 0.3,
            "name": 0.15
        }

    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"

    vessels = df["MMSI"].unique() # Get unique vessels from the original DataFrame to ensure we include all vessels, even those without gaps or name changes
    indicators = pd.DataFrame(index=vessels) # Start with all vessels as index

   

    # AIS gaps
    indicators["AIS_Gap_Count"] = gap_summary["gap_count"]

    # Route irregularity
    indicators["Route_Irregularity"] = route_irregularity_analysis(df)

    # STS
    _, sts_counts = detect_sts_events(df) 
    indicators["STS_Count"] = sts_counts

    # Name changes
    _, name_change_counts, _ = name_change_analysis(df)
    indicators["Name_Change_Count"] = name_change_counts

    indicators = indicators.fillna(0) # Fill NaNs with 0 for vessels that had no gaps, no name changes, etc.


    indicators["gap_score"]   = percentile_rank(indicators["AIS_Gap_Count"])
    indicators["route_score"] = percentile_rank(indicators["Route_Irregularity"])
    indicators["sts_score"]   = percentile_rank(indicators["STS_Count"])
    indicators["name_score"]  = percentile_rank(indicators["Name_Change_Count"])


    indicators["Risk_Score"] = (
        weights["gap"]   * indicators["gap_score"] +
        weights["sts"]   * indicators["sts_score"] +
        weights["route"] * indicators["route_score"] +
        weights["name"]  * indicators["name_score"]
    )
 

    indicators["flag_gap"]   = indicators["gap_score"]   > 0.8
    indicators["flag_route"] = indicators["route_score"] > 0.8
    indicators["flag_sts"]   = indicators["sts_score"]   > 0.8
    indicators["flag_name"]  = indicators["name_score"]  > 0.8
 
    # Number of indicators flagged (useful for explainability)
    flag_cols = ["flag_gap", "flag_route", "flag_sts", "flag_name"]
    indicators["Flag_Count"] = indicators[flag_cols].sum(axis=1)
 
    return indicators

def risk_category(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Moderate"
    elif score < 0.8:
        return "High"
    else:
        return "Extreme"
    


from sklearn.ensemble import IsolationForest

def run_anomaly_detection(indicators, contamination=0.05):
    """
    Apply Isolation Forest to detect anomalous vessels
    """

    feature_cols = [
        "gap_score",
        "route_score",
        "sts_score",
        "name_score"
    ]

    X = indicators[feature_cols]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    indicators = indicators.copy()

    indicators["anomaly_score"] = model.fit_predict(X)
    indicators["anomaly_score_raw"] = model.decision_function(X)

    # Convert to intuitive label
    indicators["is_anomalous"] = indicators["anomaly_score"] == -1

    return indicators