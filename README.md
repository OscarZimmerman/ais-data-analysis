# AIS Vessel Behaviour and Risk Analysis 

Maritime intelligence code which analyses raw AIS (Automatic Idenitification System) data and highlights/profiles vessels exhibiting illicit/anomalous behaviour using a multi-indicator risk model. 

## Context 

Financial sanctions help the UK meet its foreign policy and national security aims, as well as protecting the integrity of of its financial system. Financial sanctions include restrictions on designated persons, such as freezing financial assets, as well as wider restrictions on investment and financial services.
The UK sanctions framework derives legal authority from the Sanctions and Anti-Money Laundering Act 2018, which enables the UK to create and enforce sanctions regimes complying with international obligations and national security, independent of EU or UN models. 
 The aim of the sanctions are to undermine adversaries’ ability to wage ware, and to address malign activity as well as reinforcing enforcement across government department and agencies.
Enforcement primarily disrupts revenue stream for hostile actors - interrupting the ability of sanctioned states and networks to monetise trades and exports, which in turn disrupts and limits monetary streams into activities contrary to UK interests. It also further protects the integrity of international markets, upholding global economic restrictions and trade stability. Maritime trade routes and services are essential to the UK’s economic lifeblood. Being an island nation, effective enforcement helps protect the UK’s financial markets.  
The integration of maritime sanctions enforcement into UK economic security emerges through strategic, legal, and enforcement linkages. This role is reinforced by strategic policy, parliamentary review, and cross-government initiatives that present sanctions not only as diplomatic tools but as core components of economic security policy.

## Common evasions tactics:
**False flags and Flag hopping** - involve the misrepresentation of a vessels true sovereignty by flying under another flag. Flag hopping is repeatedly changing a flag under which the vessel is represented, causing challenges for authorities to track the vessel’s movements and activities. Although flag hopping is a common and mostly legitimate practice, attention should be given to the frequency of changes and the context of the flag state(s). 

**Ship-to-ship transfers** - STS transfers refer to the process of transferring cargo, good, or materials directly from one seafaring vessel to another while both are at sea. Can be legitimate. However, the practice of STS transfers, particularly during the night or in high risk areas, is often exploited. Commonly employed to circumvent sanctions by making the true oiling or destination of covertly transferred commodities. 

**Irregular sailing patterns** - the use of indirect routing, unscheduled detours, or transit or transhipment of cargo through third countries. Can be common, but should be scrutinised. 

**Complex ownership structures** - common practice to have complex structures for legitimate global shipping operations (to balance lawful risk exposure to assets). However, illicit actors can take advantage, including those involving shell companies and/or multiple levels of ownership and management. 

**New Vessel Acquisitions** - involves the deliberate procurement of additional ships to navigate around or mitigate the impact of imposed sanctions. May be used to obscure ownership or control of illicit actors. 

**False/fraudulent Documentation, AIS Disablement, Manipulation or Spoofing** - an Automatic Identification System is an internationally mandated system that transmits a vessel’s identification and navigational positional data via very high frequency (VHF) radio waves and satellites. Although safety issues may at times prompt legitimate disablement of AIS transmission, vessels engaged in illicit activities may go dark.  The practice of manipulating AIS data, referred to as “spoofing,” allows ships to broadcast a different name, International Maritime Organisation (IMO) number (a unique, seven-digit vessel identification code), Maritime Mobile Service Identity (MMSI), or other identifying information. This tactic can also conceal a vessel’s next port of call or other information regarding its voyage.  


## Overview

AIS transponders on ships broadcast the vessel's postition, speed, and identity at regular intervals. The project takes in data in the following pipeline:

1. Clean and validate raw AIS data
2. Compute four behavourial indicators per vessel
3. Combine indicators into weighted risk score
4. Run Isolation Forect anomaly detection
5. Stress test scores via Monte-Carlo weight sensitivity
6. Output ranked vessel summaries and visualisations


## Indicators

| Indicator | Description | Weight |
|---|---|---|
| **AIS Gap Score** | Weighted count of signal gaps: minor (10–30 min), major (30–180 min), dark (>180 min) | 35% |
| **STS Encounter Score** | Ship-to-ship meetings detected via spatial binning and haversine proximity at SOG < 2 knots | 30% |
| **Route Irregularity Score** | Standard deviation of step-distances for erratic, non-linear trajectories | 20% |
| **Name Change Score** | MMSI-linked vessel name changes over time | 15% |

Each raw indicator is converted to a percentile rank (0–1) before weighting, so the composite Risk Score is independent of dataset size, and therefore more easily comparable.


## Anomlay Detection 

An Isolation Forest (200 estimators, contamination score 5%) is run on the four normalised indicator scores, indepedent of prior weighting. Vessels which are flaggeed as anomalous by both the Indicators and the Isolation forect are treated as most likely to be illicit. 


## Monte Carlo


To test whether risk rankings hold up under different weighting assumptions, 500 random Dirichlet weight vectors are sampled and the composite score is recomputed each time. Vessels with a high `mc_mean` are robustly risky regardless of weighting. Vessels with a high `mc_std` have unstable scores and should be treated with additional caution.


## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn networkx
```

Python 3.10+ recommended.

## Usage

```bash
python main.py --data data/raw/AIS_2018_09_30.csv
```

**Options:**

| Flag | Description | Default |
|---|---|---|
| `--data` | Path to raw AIS CSV (required) | — |
| `--rows` | Max rows to load (useful for testing) | All rows |
| `--output` | Output directory for figures and CSV | `results/` |


**Example (testing on 500k rows):**
```bash
python main.py --data data/raw/AIS_2018_09_30.csv --rows 500000 --output results/
```
## Data Format

The pipeline expects a CSV with at minimum these columns:

| Column | Description |
|---|---|
| `MMSI` | Maritime Mobile Service Identity (vessel ID) |
| `BaseDateTime` | Timestamp of AIS ping |
| `LAT` | Latitude |
| `LON` | Longitude |
| `SOG` | Speed over ground (knots) |
| `VesselName` | Vessel name as broadcast |

The dataset used in development is the [NOAA/USCG AIS broadcast archive](https://marinecadastre.gov/ais/).


## Limitations and Future Work

- **Coverage bias**: vessels with few pings produce unreliable scores; a `low_coverage` flag is applied to vessels with fewer than 10 pings.
- **STS proximity threshold**: the 0.3 km threshold and 30-minute time bin are configurable but were set empirically for open-ocean data; port environments will almost definitely require tuning.
- **No ground truth**: risk scores are unsupervised and validation against known sanctioned vessel lists (e.g. OFAC SDN) would meaningfully improve calibration.
- **Limited snapshot**: the current pipeline processes a single day's data. A streaming or rolling-window version would be more operationally useful.
  

## References 

1 .  https://reeds.co.uk/insights/guide-sanctions-money-laundering-act-samla/

2.   https://assets.publishing.service.gov.uk/media/65d720cd188d770011038890/Deter-disrupt-and-demonstrate-UK-sanctions-in-a-contested-world.pdf

3.   https://committees.parliament.uk/publications/51436/documents/285625/default/

4.   https://www.gov.uk/government/publications/financial-sanctions-guidance-for-maritime-shipping/financial-sanctions-guidance-for-maritime-shipping#higher-risk-countries-and-territories

5.   https://www.nationalcrimeagency.gov.uk/who-we-are/publications/753-red-alert-shadow-fleet-sanctions-evasion-and-avoidance-network/file

6.   https://www.gov.uk/government/publications/ofsi-advisories/oil-price-cap-opc-advisory-evasion-linked-to-product-origin-manipulation-through-fabricated-and-falsified-certificates-of-origin-co
