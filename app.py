import random
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# base config
BASE_LAT = 49.0
BASE_LON = -123.0
RESTRICTED_RADIUS_KM = 15.0

random.seed(42)
np.random.seed(42)


def random_time_window(start: datetime, hours: int = 4):
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
                "valid": random.random() > 0.1,
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
    callsigns = ["FRIEND1", "FRIEND2", "CIVIL123", "UNKNOWN_X", "UNKNOWN_Y"]

    rows = []
    for _ in range(n):
        callsign = random.choice(callsigns)

        distance_km = abs(np.random.normal(loc=20.0, scale=10.0))
        bearing_deg = random.uniform(0, 360)

        dx_km = distance_km * math.cos(math.radians(bearing_deg))
        dy_km = distance_km * math.sin(math.radians(bearing_deg))

        approx_lat = BASE_LAT + (dy_km / 111.0)
        approx_lon = BASE_LON + (dx_km / 85.0)

        if callsign.startswith("CIVIL"):
            aircraft_type = random.choice(["Boeing 737", "Airbus A320", "Embraer E190"])
            airline = random.choice(["Acme Air", "Global Airways", "Coastal Air"])
            est_passengers = random.randint(50, 200)
            role = "commercial"
        elif callsign.startswith("FRIEND"):
            aircraft_type = random.choice(["Cessna 172", "Piper PA-46", "Gulfstream G450"])
            airline = ""
            est_passengers = random.randint(1, 12)
            role = "friendly"
        else:
            role = random.choice(["military", "private", "unknown"])
            if role == "military":
                aircraft_type = random.choice(["F-16", "F/A-18", "Mig-29"])
                airline = "Military"
                est_passengers = random.randint(1, 2)
            else:
                aircraft_type = random.choice(["Learjet 45", "Gulfstream G550", "Beechcraft King Air"])
                airline = ""
                est_passengers = random.randint(1, 20)

        heading_deg = bearing_deg

        rows.append(
            {
                "timestamp": random_time_window(start_time),
                "callsign": callsign,
                "lat": approx_lat,
                "lon": approx_lon,
                "altitude_ft": random.randint(1000, 15000),
                "speed_kts": random.randint(120, 480),
                "aircraft_type": aircraft_type,
                "airline": airline,
                "est_passengers": est_passengers,
                "role": role,
                "heading_deg": heading_deg,
            }
        )

    return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def classify_air_event(row, restricted_radius_km, low_alt_ft):
    distance = haversine_km(BASE_LAT, BASE_LON, row["lat"], row["lon"])

    inside_restricted = distance <= restricted_radius_km
    low_altitude = row["altitude_ft"] < low_alt_ft

    if inside_restricted and low_altitude:
        event_type = "airspace_violation"
        severity = "high"
        note = f"{row['callsign']} low altitude inside restricted zone ({distance:.1f} km)"
    elif inside_restricted:
        event_type = "airspace_entry"
        severity = "medium"
        note = f"{row['callsign']} inside restricted zone ({distance:.1f} km)"
    elif distance <= restricted_radius_km * 1.5 and low_altitude:
        event_type = "close_pass_low"
        severity = "medium"
        note = f"{row['callsign']} low altitude near boundary ({distance:.1f} km)"
    else:
        event_type = "airspace_normal"
        severity = "low"
        note = f"{row['callsign']} normal traffic ({distance:.1f} km)"

    return event_type, severity, note


def fuse_events(
    badge_df,
    camera_df,
    motion_df,
    air_df,
    tailgating_window_min,
    restricted_radius_km,
    low_alt_ft,
    motion_strong_threshold,
):
    events = []

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
                "object_id": row["person_id"],
            }
        )

    for _, row in camera_df.iterrows():
        ts = row["timestamp"]

        lookback_start = ts - timedelta(minutes=tailgating_window_min)
        recent_badges = badge_df[
            (badge_df["timestamp"] >= lookback_start)
            & (badge_df["timestamp"] <= ts)
            & (badge_df["valid"])
            & (badge_df["direction"] == "in")
        ]

        if row["alert_type"] == "person_detected" and row["zone"] == "inner_fence":
            if recent_badges.empty:
                event_type = "tailgating_suspected"
                severity = "high"
                summary = (
                    f"person at inner_fence with no valid badge entry in last {tailgating_window_min} min "
                    f"(conf {row['confidence']})"
                )
            else:
                event_type = "perimeter_person"
                severity = "medium"
                summary = f"person at inner_fence (conf {row['confidence']})"
        else:
            event_type = "camera_normal"
            severity = "low"
            summary = f"{row['alert_type']} in {row['zone']} (conf {row['confidence']})"

        events.append(
            {
                "timestamp": row["timestamp"],
                "source": "camera",
                "event_type": event_type,
                "severity": severity,
                "summary": summary,
                "object_id": row["zone"],
            }
        )

    for _, row in motion_df.iterrows():
        if row["intensity"] >= motion_strong_threshold:
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
                "object_id": row["sensor_id"],
            }
        )

    for _, row in air_df.iterrows():
        event_type, severity, note = classify_air_event(
            row,
            restricted_radius_km=restricted_radius_km,
            low_alt_ft=low_alt_ft,
        )
        events.append(
            {
                "timestamp": row["timestamp"],
                "source": "air",
                "event_type": event_type,
                "severity": severity,
                "summary": note,
                "object_id": row["callsign"],
            }
        )

    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    events_df["event_id"] = events_df.index
    return events_df


def simulate_all(restricted_radius_km, low_alt_ft, tailgating_window_min, motion_strong_threshold):
    start = datetime.now()
    badge_df = generate_badge_events(start)
    camera_df = generate_camera_alerts(start)
    motion_df = generate_motion_events(start)
    air_df = generate_air_tracks(start)
    fused_df = fuse_events(
        badge_df,
        camera_df,
        motion_df,
        air_df,
        tailgating_window_min=tailgating_window_min,
        restricted_radius_km=restricted_radius_km,
        low_alt_ft=low_alt_ft,
        motion_strong_threshold=motion_strong_threshold,
    )
    return badge_df, camera_df, motion_df, air_df, fused_df


def main():
    st.set_page_config(page_title="Base Security Fusion Console", layout="wide")
    st.title("Base + Airspace Security Fusion Console")
    st.caption("Toy Palantir-style dashboard on synthetic data")

    try:
        st.sidebar.header("Detection parameters")
        restricted_radius_km = st.sidebar.slider(
            "Restricted radius (km)",
            min_value=5,
            max_value=40,
            value=int(RESTRICTED_RADIUS_KM),
            step=1,
        )
        low_altitude_threshold_ft = st.sidebar.slider(
            "Low altitude threshold (ft)",
            min_value=1000,
            max_value=15000,
            value=5000,
            step=500,
        )
        tailgating_window_min = st.sidebar.slider(
            "Tailgating lookback (minutes)",
            min_value=2,
            max_value=30,
            value=10,
            step=1,
        )
        motion_strong_threshold = st.sidebar.slider(
            "Strong motion threshold",
            min_value=1,
            max_value=10,
            value=8,
            step=1,
        )

        badge_df, camera_df, motion_df, air_df, fused_df = simulate_all(
            restricted_radius_km=restricted_radius_km,
            low_alt_ft=low_altitude_threshold_ft,
            tailgating_window_min=tailgating_window_min,
            motion_strong_threshold=motion_strong_threshold,
        )

        # simulation time control
        min_ts = fused_df["timestamp"].min().to_pydatetime()
        max_ts = fused_df["timestamp"].max().to_pydatetime()

        st.sidebar.header("simulation time")
        current_time = st.sidebar.slider(
            "current simulated time",
            min_value=min_ts,
            max_value=max_ts,
            value=max_ts,
            format="HH:mm:ss",
        )

        visible_fused_df = fused_df[fused_df["timestamp"] <= current_time].copy()
        visible_air_df = air_df[air_df["timestamp"] <= current_time].copy()

        st.sidebar.header("filters")
        st.sidebar.caption("If the page ever appears blank, check the terminal for errors.")

        severities = sorted(visible_fused_df["severity"].unique())
        severity_selected = st.sidebar.multiselect(
            "Severity",
            options=severities,
            default=severities,
        )

        sources = sorted(visible_fused_df["source"].unique())
        source_selected = st.sidebar.multiselect(
            "Source",
            options=sources,
            default=sources,
        )

        mask = visible_fused_df["severity"].isin(severity_selected) & visible_fused_df["source"].isin(source_selected)
        filtered_df = visible_fused_df[mask].copy()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total events", len(visible_fused_df))
        with col2:
            st.metric("Medium or high", int((visible_fused_df["severity"] != "low").sum()))
        with col3:
            st.metric("High severity", int((visible_fused_df["severity"] == "high").sum()))
        with col4:
            st.metric("Airspace violations", int((visible_fused_df["event_type"] == "airspace_violation").sum()))
        with col5:
            st.metric("Tailgating suspected", int((visible_fused_df["event_type"] == "tailgating_suspected").sum()))

        alerts_df = visible_fused_df[visible_fused_df["severity"] != "low"].copy()

        st.subheader("Active alerts (medium + high)")
        if alerts_df.empty:
            st.write("no alerts in this simulation run.")
        else:
            alerts_sorted = alerts_df.sort_values("timestamp")
            st.dataframe(alerts_sorted)

            st.subheader("Alert drilldown")

            alert_ids = alerts_sorted["event_id"].tolist()

            def _label_for(eid):
                row = alerts_sorted[alerts_sorted["event_id"] == eid].iloc[0]
                return f"{int(row['event_id'])} | {row['timestamp'].strftime('%H:%M:%S')} | {row['source']} | {row['event_type']}"

            prev_sel = st.session_state.get("alert_select", None)
            try:
                default_index = alert_ids.index(prev_sel) if prev_sel in alert_ids else 0
            except Exception:
                default_index = 0

            selected_id = st.selectbox(
                "select an alert to inspect",
                options=alert_ids,
                format_func=_label_for,
                index=default_index,
                key="alert_select",
            )

            if selected_id is not None:
                selected_event = visible_fused_df[visible_fused_df["event_id"] == selected_id].iloc[0]

                window_start = selected_event["timestamp"] - timedelta(minutes=10)
                window_end = selected_event["timestamp"] + timedelta(minutes=10)

                context_df = visible_fused_df[
                    (visible_fused_df["timestamp"] >= window_start)
                    & (visible_fused_df["timestamp"] <= window_end)
                ].sort_values("timestamp")

                st.write("selected alert:")
                st.json(
                    {
                        "event_id": int(selected_event["event_id"]),
                        "timestamp": selected_event["timestamp"].isoformat(),
                        "source": selected_event["source"],
                        "event_type": selected_event["event_type"],
                        "severity": selected_event["severity"],
                        "summary": selected_event["summary"],
                    }
                )

                st.write("events in time window around alert (plus or minus 10 minutes):")
                st.dataframe(context_df)

                if selected_event["source"] == "air":
                    st.subheader("aircraft drilldown for this alert")

                    selected_callsign = selected_event["object_id"]

                    track_df = visible_air_df[visible_air_df["callsign"] == selected_callsign].sort_values("timestamp")

                    if track_df.empty:
                        st.write("no other positions for this aircraft in the current time range.")
                    else:
                        st.write(f"callsign: {selected_callsign}")
                        st.write(
                            f"track points: {len(track_df)}, "
                            f"altitude range: {int(track_df['altitude_ft'].min())}–{int(track_df['altitude_ft'].max())} ft, "
                            f"speed range: {int(track_df['speed_kts'].min())}–{int(track_df['speed_kts'].max())} kt"
                        )

                        ts_min = track_df["timestamp"].min().to_pydatetime()
                        ts_max = track_df["timestamp"].max().to_pydatetime()

                        st.write("choose a timestamp along this aircraft's path")
                        chosen_ts = st.slider(
                            "aircraft timestamp",
                            min_value=ts_min,
                            max_value=ts_max,
                            value=selected_event["timestamp"].to_pydatetime(),
                            format="HH:mm:ss",
                            key=f"air_ts_slider_{selected_callsign}_{int(selected_event['event_id'])}",
                        )

                        target_ts = pd.Timestamp(chosen_ts)
                        idx_closest = (track_df["timestamp"] - target_ts).abs().argmin()
                        current_point = track_df.iloc[idx_closest]

                        st.write(
                            f"selected point time: {current_point['timestamp'].strftime('%H:%M:%S')}, "
                            f"altitude: {int(current_point['altitude_ft'])} ft, "
                            f"speed: {int(current_point['speed_kts'])} kt"
                        )

                        st.markdown("**3d view of aircraft position and path**")

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter3d(
                                x=track_df["lon"],
                                y=track_df["lat"],
                                z=track_df["altitude_ft"],
                                mode="lines",
                                line=dict(width=2),
                                name="track",
                            )
                        )

                        role = current_point.get("role", "unknown")
                        color_map = {"commercial": "blue", "friendly": "green", "military": "red", "private": "purple", "unknown": "gray"}
                        marker_color = color_map.get(role, "black")

                        fig.add_trace(
                            go.Scatter3d(
                                x=[current_point["lon"]],
                                y=[current_point["lat"]],
                                z=[current_point["altitude_ft"]],
                                mode="markers",
                                marker=dict(size=6, color=marker_color),
                                name="current position",
                            )
                        )

                        hdg = math.radians(float(current_point["heading_deg"])) if "heading_deg" in current_point else 0.0
                        u = math.cos(hdg)
                        v = math.sin(hdg)

                        fig.add_trace(
                            go.Cone(
                                x=[current_point["lon"]],
                                y=[current_point["lat"]],
                                z=[current_point["altitude_ft"]],
                                u=[u],
                                v=[v],
                                w=[0.0],
                                sizemode="absolute",
                                sizeref=0.1,
                                anchor="tail",
                                showscale=False,
                                colorscale=[[0, marker_color], [1, marker_color]],
                                name="heading",
                            )
                        )

                        fig.update_layout(
                            scene=dict(
                                xaxis_title="longitude",
                                yaxis_title="latitude",
                                zaxis_title="altitude (ft)",
                            ),
                            margin=dict(l=0, r=0, b=0, t=0),
                            height=500,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.write("altitude over time for this aircraft")
                        st.line_chart(track_df.set_index("timestamp")[ ["altitude_ft"] ])

                        svg_map = {
                            "commercial": '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80"><rect width="200" height="80" fill="white"/><g fill="#1f77b4"><ellipse cx="60" cy="40" rx="50" ry="14"/><rect x="80" y="30" width="70" height="20" rx="6"/></g></svg>',
                            "friendly": '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80"><rect width="200" height="80" fill="white"/><g fill="#2ca02c"><ellipse cx="60" cy="40" rx="35" ry="10"/><polygon points="120,30 160,40 120,50"/></g></svg>',
                            "military": '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80"><rect width="200" height="80" fill="white"/><g fill="#d62728"><polygon points="30,40 80,20 80,60"/><rect x="80" y="34" width="70" height="12"/></g></svg>',
                            "private": '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80"><rect width="200" height="80" fill="white"/><g fill="#9467bd"><ellipse cx="60" cy="40" rx="30" ry="8"/><polygon points="100,30 130,40 100,50"/></g></svg>',
                            "unknown": '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80"><rect width="200" height="80" fill="white"/><g fill="#7f7f7f"><circle cx="60" cy="40" r="12"/><rect x="80" y="30" width="70" height="20" rx="6"/></g></svg>',
                        }

                        svg = svg_map.get(role, svg_map["unknown"]) 
                        st.image(svg, use_column_width=False, width=240)

                        st.write("flight info:")
                        st.json(
                            {
                                "callsign": selected_callsign,
                                "aircraft_type": current_point.get("aircraft_type", ""),
                                "role": role,
                                "departure": current_point.get("dep", "unknown"),
                                "destination": current_point.get("dest", "unknown"),
                            }
                        )

                        # lightweight map of raw positions
                        st.write("track positions on map")

                        # NOTE: 3D model viewer and upload/save controls have been removed per user request
                        st.info("3D model viewer removed: upload/paste/save controls disabled.")

                        # show map with track points and highlight the selected alert position
                        map_fig = go.Figure()

                        map_fig.add_trace(
                            go.Scattermapbox(
                                lat=track_df["lat"],
                                lon=track_df["lon"],
                                mode="markers+lines",
                                marker=dict(size=6, color="rgba(0,120,200,0.6)"),
                                line=dict(width=2, color="rgba(0,120,200,0.4)"),
                                name="track",
                            )
                        )

                        map_fig.add_trace(
                            go.Scattermapbox(
                                lat=[current_point["lat"]],
                                lon=[current_point["lon"]],
                                mode="markers",
                                marker=dict(size=18, color="red", opacity=0.9),
                                name="selected",
                            )
                        )

                        lat_span = track_df["lat"].max() - track_df["lat"].min()
                        lon_span = track_df["lon"].max() - track_df["lon"].min()
                        span = max(lat_span, lon_span)
                        if span < 0.01:
                            zoom = 13
                        elif span < 0.05:
                            zoom = 11
                        elif span < 0.5:
                            zoom = 9
                        else:
                            zoom = 6

                        map_fig.update_layout(
                            mapbox=dict(
                                style="open-street-map",
                                center=dict(lat=float(current_point["lat"]), lon=float(current_point["lon"])),
                                zoom=zoom,
                            ),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=350,
                        )

                        st.plotly_chart(map_fig, use_container_width=True)

        st.subheader("Event timeline (filtered)")
        st.dataframe(filtered_df.sort_values("timestamp"),)

        st.subheader("Events by source")
        source_counts = visible_fused_df["source"].value_counts()
        st.bar_chart(source_counts)

        st.subheader("Air tracks around base")
        st.caption(
            f"Base at approx lat {BASE_LAT}, lon {BASE_LON}. Restricted radius {restricted_radius_km} km "
            f"and low altitude threshold {low_altitude_threshold_ft} ft."
        )
        st.map(visible_air_df[["lat", "lon"]])
        st.subheader("How this fusion logic works")
        st.write(
            """
            this console fuses four synthetic sensor streams:
            1) badge swipes at base gates
            2) camera alerts in zones like parking_lot and inner_fence
            3) motion sensor triggers inside the base
            4) aircraft tracks flying near a restricted airspace bubble

            simple rules upgrade raw data into alerts. for example:
            - invalid badge swipes become high severity badge_invalid events
            - strong motion intensity becomes motion_strong
            - people detected at the inner fence become perimeter_person
            - if a person is seen at the inner fence and there is no valid badge entry in the last 10 minutes, a tailgating_suspected alert is raised
            - aircraft inside the restricted radius, especially at low altitude, become airspace_entry or airspace_violation
            """
        )

    except Exception as e:
        st.error("An error occurred while rendering the app.")
        st.exception(e)


if __name__ == "__main__":
    main()
