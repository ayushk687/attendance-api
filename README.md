# Employee Attendance & Analytics API

FastAPI + MongoDB (6.0+; `$densify` needs 5.1+). All code is in `app/main.py`.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MONGO_URI="mongodb://localhost:27017" MONGO_DB="attendance_db"   # or put them in a local .env
uvicorn app.main:app --port 8000
```
Indexes are created at startup (idempotent). `GET /health` pings MongoDB (503 if it is down).

## Notes
- Instants are stored as BSON dates (UTC, read back tz-aware) and exposed as epoch milliseconds.
- Attendance dates and shift times are IST; overnight shifts follow R1.
- Uniqueness (`emp_code`, and one record per employee per date) is enforced by unique indexes, not check-then-insert.
- Rates and averages are rounded half-up with `Decimal`, not with MongoDB `$round` (which is banker's rounding).
- `/admin/explain/{endpoint}` explains the same query/pipeline the endpoint runs.
