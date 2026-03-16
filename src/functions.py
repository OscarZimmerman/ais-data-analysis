import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def load_ais_data(file_path: str) -> pd.DataFrame:

    return pd.read_csv(file_path, nrows=500000)

def standardise_timestamps(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    return df

def calculate_time_del(df: pd.DataFrame, vessel_id: str, time_col: str) -> pd.DataFrame:
    
    df = df.sort_values([vessel_id, time_col])
    
    df['time_delta'] = df.groupby(vessel_id)[time_col].diff()

    return df


def ais_gap_analysis(df: pd.DataFrame, threshold: float = 24.0) -> pd.DataFrame: # in hours

    df['gap_hours'] = df['time_delta'].dt.total_seconds() / 3600.0
    df['is_gap'] = df['gap_hours'] > threshold
 
    return df

def normalise_series(series: pd.Series) -> pd.Series:
    
    return (series - series.min()) / (series.max() - series.min())

def compute_risk_score(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    
    score = 0.0
    
    for feature, weight in weights.items():
        score += weight * df[feature]

    df['risk_score'] = score

    return df

def build_vessel_port_graph(df, vessel_col, port_col):

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(row[vessel_col], row[port_col])

    return G


def compute_centrality(G):
   
    return nx.degree_centrality(G)

def assign_confidence_level(score: float) -> str:
    
    if score < 0.3:
        return "Low"
    elif score < 0.7:
        return "Moderate"
    return "High"

def plot_route(df, mmsi):
    
    vessel = df[df['MMSI'] == mmsi] # Filter for the specific vessel
    plt.figure()
    plt.plot(vessel['LON'], vessel['LAT'])
    plt.title(f"Route for vessel {mmsi}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid()
    plt.show()
