import argparse
import os
import sys

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import FancyArrowPatch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from functions import (
    ais_gap_analysis,
    compute_vessel_risk,
    detect_sts_events,
    haversine,
    name_change_analysis,
    percentile_rank,
    route_irregularity_analysis,
    run_anomaly_detection,
)





def monte_carlo_risk(indicators, n_sim=500):

    base_features = ["gap_score", "route_score", "sts_score", "name_score"]
    results = []
    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(len(base_features)))
        risk = sum(w[i] * indicators[base_features[i]] for i in range(len(base_features)))
        results.append(risk)
    mc_matrix = np.vstack(results)
    indicators = indicators.copy()
    indicators["mc_mean"] = mc_matrix.mean(axis=0)
    indicators["mc_std"] = mc_matrix.std(axis=0)
    return indicators, mc_matrix



# Pipeline steps


def load_and_preprocess(filepath: str, max_rows: int | None = None) -> pd.DataFrame:
    
    print(f"\n[1/7] Loading data from {filepath} …")
    df = pd.read_csv(filepath, nrows=max_rows)

    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.drop_duplicates(subset=["MMSI", "BaseDateTime"])
    df = df[(df["LAT"].between(-90, 90)) & (df["LON"].between(-180, 180))]
    df = df[~((df["LAT"] == 0) & (df["LON"] == 0))]

    if "SOG" in df.columns:
        df = df[df["SOG"] <= 50]

    df = df.sort_values(["MMSI", "BaseDateTime"])

    print(f"    Rows after cleaning : {len(df):,}")
    print(f"    Unique vessels      : {df['MMSI'].nunique():,}")
    return df


def run_gap_analysis(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
   
    print("\n[2/7] AIS gap analysis …")
    gap_summary = ais_gap_analysis(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    has_gaps = gap_summary[gap_summary["gap_count"] > 0]
    sns.histplot(has_gaps["gap_count"], bins=30, ax=axes[0], color="#1565C0")
    axes[0].set_title("Distribution of weighted AIS gap count")
    axes[0].set_xlabel("Weighted gap count per vessel")
    axes[0].set_ylabel("Vessel count")

    top_gap = gap_summary.nlargest(20, "gap_count")[["minor_gaps", "major_gaps", "dark_gaps"]]
    top_gap.plot(
        kind="bar", stacked=True,
        color=["#FFC107", "#FF5722", "#B71C1C"],
        ax=axes[1], width=0.8,
    )
    axes[1].set_title("Gap severity — top 20 vessels")
    axes[1].set_xlabel("MMSI")
    axes[1].set_ylabel("Gap count")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    axes[1].legend(["Minor (10–30 min)", "Major (30–180 min)", "Dark (>180 min)"])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ais_gap_analysis.png"), dpi=150)
    plt.close()

    return gap_summary


def run_risk_scoring(df: pd.DataFrame, gap_summary: pd.DataFrame) -> pd.DataFrame:

    print("\n[3/7] Computing risk scores …")
    ping_counts = df.groupby("MMSI").size().rename("ping_count")

    indicators = compute_vessel_risk(df, gap_summary=gap_summary)
    indicators = indicators.join(ping_counts)
    indicators["low_coverage"] = indicators["ping_count"] < 10

    print("\n    Top 10 vessels by risk score:")
    print(
        indicators[["Risk_Score", "Flag_Count", "ping_count", "low_coverage"]]
        .sort_values("Risk_Score", ascending=False)
        .head(10)
    )

    print("\n[4/7] Running anomaly detection (Isolation Forest) …")
    indicators = run_anomaly_detection(indicators)

    # Priority blend: risk score + anomaly signal
    indicators["priority"] = (
        indicators["Risk_Score"] * 0.6
        + indicators["anomaly_score_raw"].clip(lower=0) * 0.4
    )

    return indicators


def plot_risk_distribution(indicators: pd.DataFrame, output_dir: str) -> None:

    print("\n[5/7] Plotting risk distributions …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.kdeplot(indicators["Risk_Score"], fill=True, color="#1565C0", ax=axes[0])
    axes[0].set_title("Risk score distribution (KDE)")
    axes[0].set_xlabel("Risk score (0–1)")
    axes[0].set_ylabel("Density")

    n, bins, patches = axes[1].hist(indicators["Risk_Score"], bins=30, edgecolor="white")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdYlGn_r
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(cmap(norm(left + (bins[1] - bins[0]) / 2)))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], label="Risk score")
    axes[1].set_title("Risk score histogram")
    axes[1].set_xlabel("Risk score (0–1)")
    axes[1].set_ylabel("Count")

    plt.suptitle("Distribution of maritime behaviour risk scores", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_distribution.png"), dpi=150)
    plt.close()


def plot_network(df: pd.DataFrame, indicators: pd.DataFrame, sts_df: pd.DataFrame, output_dir: str) -> None:

    if sts_df.empty:
        print("    No STS events detected — skipping network plot.")
        return

    G = nx.Graph()
    for _, row in sts_df.iterrows():
        G.add_edge(row["MMSI1"], row["MMSI2"])

    pos = nx.spring_layout(G, seed=42)
    node_risk = [indicators.loc[n, "Risk_Score"] if n in indicators.index else 0 for n in G.nodes()]
    node_degree = [G.degree(n) for n in G.nodes()]
    node_sizes = [d * 80 + 20 for d in node_degree]

    cmap = cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx(
        G, pos,
        node_size=node_sizes,
        node_color=node_risk,
        cmap=cmap, vmin=0, vmax=1,
        edge_color="#cccccc",
        alpha=0.85,
        with_labels=False,
        ax=ax,
    )
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Risk score")
    ax.set_title("Vessel interaction network (STS)\nNode size = encounter frequency  |  Colour = risk score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sts_network.png"), dpi=150)
    plt.close()


def plot_radar(indicators: pd.DataFrame, output_dir: str) -> None:

    features = ["gap_score", "route_score", "sts_score", "name_score"]
    feature_labels = ["AIS gaps", "Route", "STS", "Name changes"]
    N = len(features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    top5 = indicators.sort_values("Risk_Score", ascending=False).head(5)
    colors = ["#B71C1C", "#E53935", "#EF9A9A", "#1565C0", "#42A5F5"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), subplot_kw=dict(polar=True))
    for ax, (mmsi, row), color in zip(axes, top5.iterrows(), colors):
        vals = [row[f] for f in features] + [row[features[0]]]
        ax.plot(angles, vals, color=color, linewidth=1.5)
        ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=6)
        ax.set_title(f"{mmsi}\nRisk: {row['Risk_Score']:.3f}", size=8, pad=12)

    plt.suptitle("Indicator profiles — top 5 vessels", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_top5.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_monte_carlo(indicators: pd.DataFrame, output_dir: str) -> None:

    print("\n[6/7] Running Monte Carlo sensitivity analysis (500 simulations) …")
    indicators, _ = monte_carlo_risk(indicators)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc = axes[0].scatter(
        indicators["mc_mean"], indicators["mc_std"],
        s=10, alpha=0.5,
        c=indicators["Risk_Score"], cmap="RdYlGn_r",
    )
    plt.colorbar(sc, ax=axes[0], label="Baseline risk score")

    unstable = indicators[(indicators["mc_mean"] > 0.6) & (indicators["mc_std"] > 0.1)]
    axes[0].scatter(
        unstable["mc_mean"], unstable["mc_std"],
        s=50, edgecolors="red", facecolors="none",
        linewidths=1.5, label=f"Unstable high-risk (n={len(unstable)})",
    )
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Mean risk score (MC)")
    axes[0].set_ylabel("Std dev of risk score")
    axes[0].set_title("Monte Carlo risk stability")

    sns.kdeplot(indicators["mc_mean"], fill=True, ax=axes[1], color="#E53935")
    axes[1].set_xlabel("Mean risk score across weight samples")
    axes[1].set_title("Distribution of MC mean risk scores")

    plt.suptitle("Monte Carlo risk score sensitivity (n=500 simulations)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monte_carlo.png"), dpi=150)
    plt.close()

    return indicators



def print_intelligence_summary(df: pd.DataFrame, indicators: pd.DataFrame) -> None:

    vessel_names = (
        df.groupby("MMSI")["VesselName"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
    )
    top5 = indicators.sort_values("Risk_Score", ascending=False).head(5)

    print("\n" + "=" * 60)
    print("VESSEL INTELLIGENCE SUMMARIES")
    print("=" * 60)

    for mmsi, row in top5.iterrows():
        fired = []
        if row["flag_gap"]:
            fired.append(f"AIS gap count in top 20% (weighted score: {int(row['AIS_Gap_Count'])})")
        if row["flag_sts"]:
            fired.append(f"STS encounter count in top 20% ({int(row['STS_Count'])} encounter(s))")
        if row["flag_route"]:
            fired.append(f"Route irregularity in top 20% (std dev: {row['Route_Irregularity']:.2f} km)")
        if row["flag_name"]:
            fired.append(f"Name changes detected ({int(row['Name_Change_Count'])} change(s))")

        anomaly_label = "flagged as anomalous" if row["is_anomalous"] else "not flagged as anomalous"
        coverage_note = " [LOW COVERAGE — treat with caution]" if row["low_coverage"] else ""
        vessel_name = vessel_names.get(mmsi, "Unknown")

        print(f"\nMMSI {mmsi}  |  {vessel_name}{coverage_note}")
        print(f"  Risk score  : {row['Risk_Score']:.3f}  |  Flags fired: {int(row['Flag_Count'])}/4")
        print(f"  Anomaly     : {anomaly_label} (score: {row['anomaly_score_raw']:.3f})")
        print(f"  Priority    : {row['priority']:.3f}")
        if fired:
            print("  Reasons:")
            for f in fired:
                print(f"    · {f}")
        else:
            print("  Reasons: no individual flags fired (elevated composite score)")

    print("\n" + "=" * 60)


# Entry point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIS vessel risk analysis pipeline")
    parser.add_argument("--data", required=True, help="Path to raw AIS CSV file")
    parser.add_argument("--rows", type=int, default=None, help="Max rows to load (default: all)")
    parser.add_argument("--output", default="results", help="Directory for output files (default: results/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    df = load_and_preprocess(args.data, max_rows=args.rows)
    gap_summary = run_gap_analysis(df, args.output)

    print("\n    Running STS detection …")
    sts_df, _ = detect_sts_events(df)

    indicators = run_risk_scoring(df, gap_summary)

    plot_risk_distribution(indicators, args.output)
    plot_network(df, indicators, sts_df, args.output)
    plot_radar(indicators, args.output)
    indicators = plot_monte_carlo(indicators, args.output)

    print_intelligence_summary(df, indicators)

    out_csv = os.path.join(args.output, "vessel_risk_scores.csv")
    indicators.to_csv(out_csv)
    print(f"\nRisk scores saved → {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()