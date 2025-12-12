import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import math

# base config
BASE_LAT = 49.0
BASE_LON = -123.0
RESTRICTED_RADIUS_KM = 15.0

random.seed(42)
np.random.seed(42)

def random_time_window(start: datetime, hours: int = 4):
    """Return a random datetime in the next `hours` hours."""
    return start + timedelta(minutes=random.randint(0, hours * 60))

def generate_badge_events(start_time: datetime, n=20):
    people = ["alice", "bob", "charlie", "dana"]
    gates = ["north_gate", "south_gate", "east_gate"]

    rows = []
    for _ in range(n):
        rows.append(
            {
                "timestamp": random_time_window(start_time),
                "person_id": random.choice(people),
                "gate": random.choice(gates),
                "direction": random.choice(["in", "out"]),
                "valid": random.random() > 0.1,  # 10% invalid swipes
            }
        )
    return pd.DataFrame(rows)

def generate_camera_alerts(start_time: datetime, n=15):
    zones = ["parking_lot", "inner_fence", "hq_corridor"]
    alert_types = ["person_detected", "vehicle_detected"]

    rows = []
    for _ in range(n):
        rows.append(
            {
                "timestamp": random_time_window(start_time),
                "zone": random.choice(zones),
                "alert_type": random.choice(alert_types),
                "confidence": round(random.uniform(0.6, 0.99), 2),
            }
        )
    return pd.DataFrame(rows)

def generate_motion_events(start_time: datetime, n=15):
    sensors = ["motion_A1", "motion_B2", "motion_C3"]

    rows = []
    for _ in range(n):
        rows.append(
            {
                "timestamp": random_time_window(start_time),
                "sensor_id": random.choice(sensors),
                "intensity": random.randint(1, 10),
            }
        )
    return pd.DataFrame(rows)

def generate_air_tracks(start_time: datetime, n=25):
    """
    Simulate aircraft positions near the base.
    Some will pass through the restricted radius.
    """
    callsigns = ["FRIEND1", "FRIEND2", "CIVIL123", "UNKNOWN_X", "UNKNOWN_Y"]

    rows = []
    for _ in range(n):
        callsign = random.choice(callsigns)

        # distance from base in km
        distance_km = np.abs(np.random.normal(loc=20.0, scale=10.0))
        bearing_deg = random.uniform(0, 360)

        # simple polar to cartesian approximation (ignoring earth curvature)
        dx_km = distance_km * np.cos(np.radians(bearing_deg))
        dy_km = distance_km * np.sin(np.radians(bearing_deg))

        approx_lat = BASE_LAT + (dy_km / 111.0)
        approx_lon = BASE_LON + (dx_km / 85.0)  # rough correction

        rows.append(
            {
                "timestamp": random_time_window(start_time),
                "callsign": callsign,
                "lat": approx_lat,
                "lon": approx_lon,
                "altitude_ft": random.randint(1000, 15000),
                "speed_kts": random.randint(120, 480),
            }
        )

    return pd.DataFrame(rows)

def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate distance in km between two lat/lon points."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def classify_air_event(row):
    """Label air track as normal or violation based on distance and altitude."""
    distance = haversine_km(BASE_LAT, BASE_LON, row["lat"], row["lon"])

    inside_restricted = distance <= RESTRICTED_RADIUS_KM
    low_altitude = row["altitude_ft"] < 5000

    if inside_restricted and low_altitude:
        event_type = "airspace_violation"
        severity = "high"
        note = f"{row['callsign']} low altitude inside restricted zone ({distance:.1f} km)"
    elif inside_restricted:
        event_type = "airspace_entry"
        severity = "medium"
        note = f"{row['callsign']} inside restricted zone ({distance:.1f} km)"
    elif distance <= RESTRICTED_RADIUS_KM * 1.5 and low_altitude:
        event_type = "close_pass_low"
        severity = "medium"
        note = f"{row['callsign']} low altitude near boundary ({distance:.1f} km)"
    else:
        event_type = "airspace_normal"
        severity = "low"
        note = f"{row['callsign']} normal traffic ({distance:.1f} km)"

    return event_type, severity, note

def fuse_events(badge_df, camera_df, motion_df, air_df):
    """turn raw sensor tables into a unified event timeline."""
    events = []

    # badge events
    for _, row in badge_df.iterrows():
        if row["valid"]:
            event_type = "badge_normal"
            severity = "low"
        else:
            event_type = "badge_invalid"
            severity = "high"

        events.append(
            {
                "timestamp": row["timestamp"],
                "source": "badge",
                "event_type": event_type,
                "severity": severity,
                "summary": f"badge {row['direction']} by {row['person_id']} at {row['gate']}",
            }
        )

    # camera alerts
    for _, row in camera_df.iterrows():
        if row["alert_type"] == "person_detected" and row["zone"] == "inner_fence":
            event_type = "perimeter_person"
            severity = "medium"
        else:
            event_type = "camera_normal"
            severity = "low"

        events.append(
            {
                "timestamp": row["timestamp"],
                "source": "camera",
                "event_type": event_type,
                "severity": severity,
                "summary": f"{row['alert_type']} in {row['zone']} (conf {row['confidence']})",
            }
        )

    # motion sensors
    for _, row in motion_df.iterrows():
        if row["intensity"] >= 8:
            event_type = "motion_strong"
            severity = "medium"
        else:
            event_type = "motion_normal"
            severity = "low"

        events.append(
            {
                "timestamp": row["timestamp"],
                "source": "motion",
                "event_type": event_type,
                "severity": severity,
                "summary": f"motion {row['sensor_id']} intensity {row['intensity']}",
            }
        )

    # air tracks
    for _, row in air_df.iterrows():
        event_type, severity, note = classify_air_event(row)

        events.append(
            {
                "timestamp": row["timestamp"],
                "source": "air",
                "event_type": event_type,
                "severity": severity,
                "summary": note,
            }
        )

    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    return events_df

def main():
    start = datetime.now()
    badge_df = generate_badge_events(start)
    camera_df = generate_camera_alerts(start)
    motion_df = generate_motion_events(start)
    air_df = generate_air_tracks(start)

    print("badge events:")
    print(badge_df.head(), "\n")

    print("camera alerts:")
    print(camera_df.head(), "\n")

    print("motion events:")
    print(motion_df.head(), "\n")

    print("air tracks:")
    print(air_df.head(), "\n")

    fused_df = fuse_events(badge_df, camera_df, motion_df, air_df)

    print("unified event timeline (first 10):")
    print(fused_df.head(10), "\n")

    print("non low severity events:")
    print(fused_df[fused_df["severity"] != "low"]) 

if __name__ == "__main__":
    main()
