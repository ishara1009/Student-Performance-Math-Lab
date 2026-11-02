# Dataset

Place the supplied synthetic CSV at `data/raw/student_performance_sample.csv`. It must contain a
unique `student_id`, weekly `study_hours_per_week` (0–168), `attendance_pct` (0–100), daily
`sleep_hours` (0–24), `extra_classes` (`Yes`/`No`), and `final_score` (0–100).

The four non-ID inputs are model features and `final_score` is the target. `student_id` is never a
feature. The data's exact generation process is not recorded here because it was not supplied.

