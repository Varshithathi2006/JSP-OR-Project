# Multi-objective Taillard JSP

This project models the Taillard job-shop scheduling problem using a baseline MILP, weighted goal programming, benchmark targets, weight sensitivity analysis, and Gantt-chart output.

The project also includes `app.py`, a Streamlit visual solving tool using only the local Taillard dataset.

## Setup

```powershell
py -3 -m pip install -r requirements.txt
```

## Launch the visual tool

```powershell
py -3 -m streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`. Select one of `ta01` to `ta80`, choose an objective, and click **Solve selected instance**.

For deployment, upload this folder to a GitHub repository and select `app.py` as the main file in Streamlit Community Cloud. The dataset is bundled under `jsp/taillard`, so no external data service is required.

## Run on ta01

```powershell
py -3 jsp_solver.py jsp\taillard\ta01 --mode makespan --time-limit 120 --reference-makespan 1231 --output-dir outputs

# Add --sensitivity to run the weighted goal-programming scenarios and write the CSV.
py -3 jsp_solver.py jsp\taillard\ta01 --mode goal --time-limit 120 --sensitivity --output-dir outputs
```

The command prints schedule metrics and the reference gap, then writes a Gantt chart plus sensitivity CSV into `outputs`. Reference values should come from the same Taillard file convention used by the local dataset.

The Taillard files use zero-based machine IDs. The chart labels them as `M1`, `M2`, and so on for readability.

## Model notes

For each pair of operations sharing a machine, one binary variable selects their order. The Big-M value is the sum of all processing times, which is a valid scheduling horizon. Flow time is the sum of job completion times. Idle time is measured over the makespan horizon as `machines * Cmax - total processing time`; for a fixed instance it is therefore mathematically linked to makespan and should be interpreted accordingly in the sensitivity report.