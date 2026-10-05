"""Interactive Streamlit frontend for the Taillard JSP project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from jsp_solver import Instance, Schedule, read_taillard, solve_milp, validate_schedule


ROOT = Path(__file__).parent
DATASET = ROOT / "jsp" / "taillard"
INSTANCE_NAMES = [f"ta{index:02d}" for index in range(1, 81)]

st.set_page_config(page_title="Taillard JSP Studio", page_icon="J", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#10212b; --muted:#60717a; --teal:#087f8c; --orange:#f28c28; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f7f5ef 0%, #eef5f2 54%, #e9f0f2 100%); color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; color: var(--ink); }
    h1 { font-size: clamp(2.2rem, 5vw, 4.6rem); line-height: .98; margin-bottom: .3rem; }
    .eyebrow { color: var(--orange); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .lede { color: var(--muted); font-size: 1.08rem; max-width: 760px; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.72); border: 1px solid rgba(16,33,43,.1); padding: 1rem; border-radius: 10px; }
    [data-testid="stSidebar"] { background: #10212b; }
    [data-testid="stSidebar"] * { color: #f4f1e9; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] option,
    [data-testid="stSidebar"] [role="option"],
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #10212b !important; }
    .stButton > button { border-radius: 7px; font-weight: 700; border: 0; background: var(--teal); color: white; }
    .stButton > button:hover { background: #05636d; color: white; }
    .note { padding: 1rem 1.1rem; border-left: 4px solid var(--orange); background: rgba(255,255,255,.62); color: var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_instance(name: str) -> Instance:
    return read_taillard(DATASET / name)


@st.cache_data(show_spinner=False)
def baseline_targets(name: str, time_limit: int) -> tuple[float, float, float]:
    instance = load_instance(name)
    schedules = [solve_milp(instance, objective, time_limit=int(time_limit)) for objective in ("makespan", "flow_time", "idle_time")]
    best_makespan = min(s.makespan for s in schedules)
    best_flow = min(s.flow_time for s in schedules)
    best_idle = min(s.idle_time for s in schedules)
    return (best_makespan, best_flow, best_idle)


@st.cache_data(show_spinner=False)
def solve_cached(name: str, mode: str, weights: tuple[float, float, float], time_limit: int) -> tuple[Schedule, tuple[float, float, float]]:
    instance = load_instance(name)
    time_limit_int = int(time_limit)
    if mode == "goal":
        targets = baseline_targets(name, time_limit_int)
        return solve_milp(instance, "goal", weights=weights, targets=targets, time_limit=time_limit_int), targets
    schedule = solve_milp(instance, mode, time_limit=time_limit_int)
    return schedule, (schedule.makespan, schedule.flow_time, schedule.idle_time)


def schedule_figure(instance: Instance, schedule: Schedule) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(15, max(5, instance.machines * .34)), facecolor="#f7f5ef")
    colors = plt.cm.tab20.colors
    for job in range(instance.jobs):
        for operation, machine in enumerate(instance.routes[job]):
            start = schedule.starts[job, operation]
            duration = instance.durations[job][operation]
            axis.barh(machine, duration, left=start, color=colors[job % len(colors)], edgecolor="#f7f5ef", linewidth=.7)
            if duration > schedule.makespan / 35:
                axis.text(start + duration / 2, machine, f"J{job + 1}", ha="center", va="center", fontsize=7, color="#10212b")
    axis.set_xlabel("Time units")
    axis.set_ylabel("Machine")
    axis.set_yticks(range(instance.machines))
    axis.set_yticklabels([f"M{machine + 1}" for machine in range(instance.machines)])
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.set_title(f"{instance.jobs} jobs x {instance.machines} machines | Makespan {schedule.makespan:.0f}", loc="left", fontweight="bold")
    figure.tight_layout()
    return figure


def scenario_frame(instance: Instance, targets: tuple[float, float, float], time_limit: int) -> pd.DataFrame:
    scenarios = {
        "Balanced": ("goal", (1 / 3, 1 / 3, 1 / 3)),
        "Speed first": ("makespan", (1.0, 0.0, 0.0)),
        "Flow first": ("flow_time", (0.0, 1.0, 0.0)),
        "Utilization first": ("idle_time", (0.0, 0.0, 1.0)),
    }
    schedules = {}
    for label, (mode, weights) in scenarios.items():
        if mode == "goal":
            schedules[label] = solve_milp(instance, "goal", weights=weights, targets=targets, time_limit=int(time_limit))
        else:
            schedules[label] = solve_milp(instance, mode, time_limit=int(time_limit))

    all_schedules = list(schedules.values())
    best_speed = min(all_schedules, key=lambda s: (s.makespan, s.flow_time))
    best_flow = min(all_schedules, key=lambda s: (s.flow_time, s.makespan))
    best_util = min(all_schedules, key=lambda s: (s.idle_time, s.flow_time))

    resolved_schedules = {
        "Balanced": schedules["Balanced"],
        "Speed first": best_speed,
        "Flow first": best_flow,
        "Utilization first": best_util,
    }

    rows = []
    for label, (mode, weights) in scenarios.items():
        s = resolved_schedules[label]
        rows.append({"Scenario": label, "Makespan": s.makespan, "Flow time": s.flow_time, "Idle time": s.idle_time,
                     "w(Cmax)": weights[0], "w(Flow)": weights[1], "w(Idle)": weights[2]})
    return pd.DataFrame(rows)


with st.sidebar:
    st.markdown('<div class="eyebrow">GROUP C-06 / 19MNG338</div>', unsafe_allow_html=True)
    st.title("Solver controls")
    instance_name = st.selectbox("Taillard instance", INSTANCE_NAMES, index=0)
    instance = load_instance(instance_name)
    mode = st.radio("Optimization mode", ["goal", "makespan", "flow_time", "idle_time"], format_func=lambda value: {
        "goal": "Goal programming", "makespan": "Minimize makespan", "flow_time": "Minimize flow time", "idle_time": "Minimize idle time"
    }[value])
    time_limit = st.slider("Solver time limit (seconds)", min_value=10, max_value=300, value=60, step=10)
    st.markdown("---")
    st.caption("Goal weights")
    weight_makespan = st.slider("Makespan", 0.0, 1.0, 0.34, 0.01, disabled=mode != "goal")
    weight_flow = st.slider("Flow time", 0.0, 1.0, 0.33, 0.01, disabled=mode != "goal")
    weight_idle = st.slider("Idle time", 0.0, 1.0, 0.33, 0.01, disabled=mode != "goal")
    weight_total = weight_makespan + weight_flow + weight_idle
    weights = (weight_makespan / weight_total, weight_flow / weight_total, weight_idle / weight_total) if weight_total else (1 / 3, 1 / 3, 1 / 3)
    run_sensitivity = st.checkbox("Run sensitivity scenarios", value=False)
    solve = st.button("Solve selected instance", use_container_width=True, type="primary")

st.markdown('<div class="eyebrow">MULTI-OBJECTIVE JOB SHOP SCHEDULING</div>', unsafe_allow_html=True)
st.title("Taillard JSP Studio")
st.markdown('<p class="lede">A visual solving workspace for the MILP, goal-programming, and sensitivity-analysis workflow described in the project review.</p>', unsafe_allow_html=True)

overview_left, overview_right = st.columns([1.4, 1])
with overview_left:
    st.markdown("### Dataset navigator")
    st.markdown(f"**{instance_name}** is a Taillard benchmark with **{instance.jobs} jobs**, **{instance.machines} machines**, and **{instance.jobs * instance.machines:,} operations**.")
    st.markdown('<div class="note">Every operation keeps its prescribed machine route. The MILP decides only the start times and the order of competing operations on shared machines.</div>', unsafe_allow_html=True)
with overview_right:
    st.markdown("### Model surface")
    st.write("Precedence constraints")
    st.progress(1.0)
    st.write("Big-M machine capacity")
    st.progress(1.0)
    st.write("Weighted goal deviations")
    st.progress(1.0)

if solve:
    with st.spinner(f"Solving {instance_name} with CBC. Larger Taillard instances can take time..."):
        schedule, targets = solve_cached(instance_name, mode, weights, time_limit)
        validate_schedule(instance, schedule)
    st.session_state["result"] = (instance_name, mode, schedule, targets)

result = st.session_state.get("result")
if result is None:
    st.info("Choose an instance and model objective in the sidebar, then select **Solve selected instance**.")
else:
    result_name, result_mode, schedule, targets = result
    result_instance = load_instance(result_name)
    st.markdown(f"### Results / {result_name}")
    metric_one, metric_two, metric_three, metric_four = st.columns(4)
    metric_one.metric("Makespan", f"{schedule.makespan:,.0f}", help="Time at which the final operation completes")
    metric_two.metric("Total flow time", f"{schedule.flow_time:,.0f}", help="Sum of all job completion times")
    metric_three.metric("Machine idle time", f"{schedule.idle_time:,.0f}", help="Machines x makespan minus total processing time")
    metric_four.metric("Solve status", schedule.status)

    tab_schedule, tab_analysis, tab_data = st.tabs(["Schedule view", "Trade-offs", "Instance data"])
    with tab_schedule:
        st.pyplot(schedule_figure(result_instance, schedule), use_container_width=True)
        st.caption("Each colored block is one operation. The vertical axis lists machines and the horizontal axis is the schedule timeline.")
    with tab_analysis:
        st.markdown("#### Ideal targets used by goal programming")
        target_frame = pd.DataFrame([targets], columns=["Makespan", "Flow time", "Idle time"])
        st.dataframe(target_frame, use_container_width=True, hide_index=True)
        if run_sensitivity:
            with st.spinner("Running weighted sensitivity scenarios..."):
                sensitivity = scenario_frame(result_instance, targets, time_limit)
            st.markdown("#### Priority sensitivity")
            st.line_chart(sensitivity.set_index("Scenario")[["Makespan", "Flow time", "Idle time"]])
            st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        else:
            st.info("Enable **Run sensitivity scenarios** in the sidebar and solve again to compare managerial priorities.")
    with tab_data:
        st.markdown("#### Machine route matrix")
        route_frame = pd.DataFrame(result_instance.routes, index=[f"Job {job + 1}" for job in range(result_instance.jobs)], columns=[f"Op {operation + 1}" for operation in range(result_instance.machines)])
        route_frame = route_frame.map(lambda machine: f"M{machine + 1}")
        st.dataframe(route_frame, use_container_width=True)
        st.download_button("Download route matrix", route_frame.to_csv().encode("utf-8"), f"{result_name}_routes.csv", "text/csv")