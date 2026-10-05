"""MILP and weighted goal-programming solver for Taillard job-shop instances."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pulp


@dataclass(frozen=True)
class Instance:
    jobs: int
    machines: int
    routes: list[list[int]]
    durations: list[list[int]]


@dataclass
class Schedule:
    starts: dict[tuple[int, int], float]
    makespan: float
    flow_time: float
    idle_time: float
    status: str


def read_taillard(path: str | Path) -> Instance:
    """Read a Taillard file containing machine/time pairs."""
    values = [int(token) for token in Path(path).read_text(encoding="utf-8").split()]
    if len(values) < 2:
        raise ValueError(f"{path} does not contain a jobs/machines header")
    jobs, machines = values[:2]
    expected = 2 + jobs * machines * 2
    if len(values) != expected:
        raise ValueError(f"{path} contains {len(values) - 2} operation values; expected {expected - 2}")
    routes: list[list[int]] = []
    durations: list[list[int]] = []
    cursor = 2
    for _ in range(jobs):
        job_routes: list[int] = []
        job_durations: list[int] = []
        for _ in range(machines):
            job_routes.append(values[cursor])
            job_durations.append(values[cursor + 1])
            cursor += 2
        routes.append(job_routes)
        durations.append(job_durations)
    return Instance(jobs, machines, routes, durations)


def _operation_pairs(instance: Instance) -> Iterable[tuple[tuple[int, int], tuple[int, int]]]:
    by_machine: dict[int, list[tuple[int, int]]] = {machine: [] for machine in range(instance.machines)}
    for job in range(instance.jobs):
        for operation, machine in enumerate(instance.routes[job]):
            by_machine[machine].append((job, operation))
    for operations in by_machine.values():
        for position, first in enumerate(operations):
            for second in operations[position + 1 :]:
                yield first, second


def solve_milp(
    instance: Instance,
    objective: str = "makespan",
    weights: tuple[float, float, float] | None = None,
    targets: tuple[float, float, float] | None = None,
    time_limit: int | None = None,
) -> Schedule:
    """Solve a single objective or weighted goal-programming model."""
    if objective not in {"makespan", "flow_time", "idle_time", "goal"}:
        raise ValueError("objective must be makespan, flow_time, idle_time, or goal")
    if objective == "goal" and (weights is None or targets is None):
        raise ValueError("goal programming requires weights and targets")

    jobs = int(instance.jobs)
    machines = int(instance.machines)
    time_limit_val = int(time_limit) if time_limit is not None else None

    horizon = sum(sum(job) for job in instance.durations)
    model = pulp.LpProblem("taillard_jsp", pulp.LpMinimize)
    starts = {
        (job, operation): pulp.LpVariable(f"s_{job}_{operation}", 0)
        for job in range(jobs)
        for operation in range(machines)
    }
    makespan = pulp.LpVariable("c_max", 0)
    completion = {
        job: starts[job, machines - 1] + instance.durations[job][-1]
        for job in range(jobs)
    }

    for job in range(jobs):
        for operation in range(machines - 1):
            model += starts[job, operation + 1] >= starts[job, operation] + instance.durations[job][operation]
        model += makespan >= completion[job]

    for index, (first, second) in enumerate(_operation_pairs(instance)):
        first_job, first_operation = first
        second_job, second_operation = second
        order = pulp.LpVariable(f"order_{index}", 0, 1, cat=pulp.LpBinary)
        first_end = starts[first] + instance.durations[first_job][first_operation]
        second_end = starts[second] + instance.durations[second_job][second_operation]
        model += starts[second] >= first_end - horizon * (1 - order)
        model += starts[first] >= second_end - horizon * order

    flow_time = pulp.lpSum(completion.values())
    idle_time = machines * makespan - sum(sum(job) for job in instance.durations)
    if objective == "makespan":
        model += makespan
    elif objective == "flow_time":
        model += flow_time
    elif objective == "idle_time":
        model += idle_time
    else:
        deviation_over = [pulp.LpVariable(f"d_plus_{index}", 0) for index in range(3)]
        deviation_under = [pulp.LpVariable(f"d_minus_{index}", 0) for index in range(3)]
        objectives = [makespan, flow_time, idle_time]
        for index, expression in enumerate(objectives):
            model += expression + deviation_under[index] - deviation_over[index] == targets[index]
        model += pulp.lpSum(weights[index] * deviation_over[index] / max(targets[index], 1) for index in range(3))

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_val)
    model.solve(solver)
    status = pulp.LpStatus[model.status]
    if status not in {"Optimal", "Feasible"}:
        raise RuntimeError(f"MILP solve failed with status {status}")

    solved_starts = {key: float(pulp.value(variable)) for key, variable in starts.items()}
    solved_makespan = float(pulp.value(makespan))
    solved_flow_time = sum(solved_starts[job, machines - 1] + instance.durations[job][-1] for job in range(jobs))
    solved_idle_time = machines * solved_makespan - sum(sum(job) for job in instance.durations)
    return Schedule(solved_starts, solved_makespan, solved_flow_time, solved_idle_time, status)


def solve_goal_programming(instance: Instance, weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3), time_limit: int | None = None) -> tuple[Schedule, tuple[float, float, float]]:
    """Compute ideal single-objective targets, then solve weighted GP."""
    target_schedules = [solve_milp(instance, objective, time_limit=time_limit) for objective in ("makespan", "flow_time", "idle_time")]
    best_makespan = min(s.makespan for s in target_schedules)
    best_flow = min(s.flow_time for s in target_schedules)
    best_idle = min(s.idle_time for s in target_schedules)
    targets = (best_makespan, best_flow, best_idle)
    return solve_milp(instance, "goal", weights=weights, targets=targets, time_limit=time_limit), targets


def validate_schedule(instance: Instance, schedule: Schedule, tolerance: float = 1e-6) -> None:
    """Raise ValueError if precedence or machine-capacity constraints are violated."""
    for job in range(instance.jobs):
        for operation in range(instance.machines - 1):
            current_end = schedule.starts[job, operation] + instance.durations[job][operation]
            if schedule.starts[job, operation + 1] + tolerance < current_end:
                raise ValueError(f"precedence violation for job {job + 1}")
    for first, second in _operation_pairs(instance):
        first_end = schedule.starts[first] + instance.durations[first[0]][first[1]]
        second_end = schedule.starts[second] + instance.durations[second[0]][second[1]]
        if schedule.starts[second] + tolerance < first_end and schedule.starts[first] + tolerance < second_end:
            raise ValueError(f"machine overlap between {first} and {second}")


def plot_gantt(instance: Instance, schedule: Schedule, output: str | Path) -> None:
    """Write a machine-oriented Gantt chart for a solved schedule."""
    figure, axis = plt.subplots(figsize=(14, max(5, instance.machines * 0.35)))
    colors = plt.cm.tab20.colors
    for job in range(instance.jobs):
        for operation, machine in enumerate(instance.routes[job]):
            start = schedule.starts[job, operation]
            duration = instance.durations[job][operation]
            axis.barh(machine, duration, left=start, color=colors[job % len(colors)], edgecolor="white")
            axis.text(start + duration / 2, machine, f"J{job + 1}", ha="center", va="center", fontsize=7)
    axis.set_xlabel("Time")
    axis.set_ylabel("Machine")
    axis.set_title(f"Taillard schedule | Cmax={schedule.makespan:.0f}, flow={schedule.flow_time:.0f}")
    axis.set_yticks(range(instance.machines))
    axis.set_yticklabels([f"M{machine + 1}" for machine in range(instance.machines)])
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_sensitivity(instance: Instance, output: str | Path, time_limit: int | None = None) -> None:
    """Run balanced and single-priority weight scenarios."""
    scenarios = {
        "balanced": (1 / 3, 1 / 3, 1 / 3),
        "makespan": (1.0, 0.0, 0.0),
        "flow_time": (0.0, 1.0, 0.0),
        "idle_time": (0.0, 0.0, 1.0),
    }
    rows = []
    for name, weights in scenarios.items():
        schedule, targets = solve_goal_programming(instance, weights, time_limit)
        rows.append({
            "scenario": name,
            "w_makespan": weights[0], "w_flow_time": weights[1], "w_idle_time": weights[2],
            "target_makespan": targets[0], "target_flow_time": targets[1], "target_idle_time": targets[2],
            "makespan": schedule.makespan, "flow_time": schedule.flow_time, "idle_time": schedule.idle_time,
        })
    with Path(output).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Taillard JSP instances with MILP or goal programming.")
    parser.add_argument("instance", type=Path)
    parser.add_argument("--mode", choices=("makespan", "flow_time", "idle_time", "goal"), default="goal")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--reference-makespan", type=float, default=None, help="Published reference value for gap reporting")
    parser.add_argument("--sensitivity", action="store_true", help="Run the four weighted goal-programming scenarios")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instance = read_taillard(args.instance)
    if args.mode == "goal":
        schedule, targets = solve_goal_programming(instance, time_limit=args.time_limit)
        print(f"Targets: makespan={targets[0]:.0f}, flow={targets[1]:.0f}, idle={targets[2]:.0f}")
    else:
        schedule = solve_milp(instance, args.mode, time_limit=args.time_limit)
    validate_schedule(instance, schedule)
    stem = args.instance.stem
    plot_gantt(instance, schedule, args.output_dir / f"{stem}_{args.mode}_gantt.png")
    if args.sensitivity:
        run_sensitivity(instance, args.output_dir / f"{stem}_sensitivity.csv", args.time_limit)
    print(f"Status: {schedule.status}")
    print(f"Makespan: {schedule.makespan:.0f} | Flow time: {schedule.flow_time:.0f} | Idle time: {schedule.idle_time:.0f}")
    if args.reference_makespan is not None:
        gap = 100 * (schedule.makespan - args.reference_makespan) / args.reference_makespan
        print(f"Reference makespan: {args.reference_makespan:.0f} | Gap: {gap:.2f}%")


if __name__ == "__main__":
    main()