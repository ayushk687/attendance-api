"""
Employee Attendance & Analytics API (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (real environment variables win over a local .env)
"""
import json
import os
import re
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

# --- BEGIN PURE HELPERS (stdlib only, unit-testable without MongoDB) ------------------------------------------ #
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))  # India has no DST
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]  # R6
MS_MIN, MS_MAX = 100_000_000_000, 4_102_444_800_000


def as_utc(dt: datetime) -> datetime:
    """PyMongo can hand back naive datetimes; they are always UTC. Make everything tz-aware."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def trunc(dt: datetime) -> datetime:
    """R1: instants are truncated to whole seconds before any calculation."""
    return as_utc(dt).replace(microsecond=0)


def ms_to_dt(ms: int) -> datetime:
    return trunc(EPOCH + timedelta(milliseconds=ms))


def dt_to_ms(dt: datetime) -> int:
    return (as_utc(dt) - EPOCH) // timedelta(milliseconds=1)


def opt_ms(dt) -> Optional[int]:
    return None if dt is None else dt_to_ms(dt)


def half_up(value, places: int) -> float:
    """R4/R8: round half-up (never banker's rounding, never float rounding)."""
    d = value if isinstance(value, Decimal) else Decimal(str(value))
    return float(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def is_overnight(emp: dict) -> bool:
    return emp["shift_end"] <= emp["shift_start"]  # zero-padded HH:MM compares correctly as text


def attendance_date(punch_in: datetime, emp: dict) -> str:
    """R1: IST date of the punch-in; an overnight-shift punch-in earlier than shift_end belongs to the previous day."""
    ist = as_utc(punch_in).astimezone(IST)
    d = ist.date()
    if is_overnight(emp) and ist.time() < hhmm(emp["shift_end"]):
        d -= timedelta(days=1)
    return d.isoformat()


def shift_start_dt(date_str: str, emp: dict) -> datetime:
    return datetime.combine(date.fromisoformat(date_str), hhmm(emp["shift_start"]), tzinfo=IST)


def shift_end_dt(date_str: str, emp: dict) -> datetime:
    d = date.fromisoformat(date_str) + (timedelta(days=1) if is_overnight(emp) else timedelta(0))
    return datetime.combine(d, hhmm(emp["shift_end"]), tzinfo=IST)


def late_minutes_for(punch_in: datetime, date_str: str, emp: dict) -> int:
    """R2: late only if STRICTLY more than 10:00 after shift_start; then floor(minutes since shift_start)."""
    elapsed = (punch_in - shift_start_dt(date_str, emp)).total_seconds()
    return int(elapsed // 60) if elapsed > 600 else 0


def overtime_for(punch_out: datetime, date_str: str, emp: dict) -> int:
    """R3: floor(minutes after shift_end), counted only when >= 30."""
    diff = (punch_out - shift_end_dt(date_str, emp)).total_seconds()
    minutes = int(diff // 60) if diff > 0 else 0
    return minutes if minutes >= 30 else 0


def work_hours_for(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600, half-up to 2 decimals."""
    return half_up(Decimal(int((punch_out - punch_in).total_seconds())) / Decimal(3600), 2)


def derive(emp: dict, status: str, punch_in, punch_out, date_str: str) -> dict:
    """The four derived fields for a record, per DATA_MODEL.md section 3."""
    if status not in PRESENCE or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    late = late_minutes_for(punch_in, date_str, emp)
    if punch_out is None:
        return {"work_hours": None, "late_minutes": late, "overtime_minutes": 0, "half_day": False}
    wh = work_hours_for(punch_in, punch_out)
    return {
        "work_hours": wh,
        "late_minutes": late,
        "overtime_minutes": overtime_for(punch_out, date_str, emp),
        "half_day": wh < 4.50,  # R5: uses the ROUNDED work_hours
    }


def valid_date(s: str) -> bool:
    try:
        return bool(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", s)) and bool(date.fromisoformat(s))
    except ValueError:
        return False


def month_bounds(month: str):
    y, m = int(month[:4]), int(month[5:7])
    first = date(y, m, 1)
    last = (date(y + (m == 12), m % 12 + 1, 1)) - timedelta(days=1)
    return first.isoformat(), last.isoformat()


def working_days(start: str, end: str, joined_on: str) -> int:
    """R7: Mon-Fri, counted from joined_on when the employee joined mid-period."""
    d = date.fromisoformat(max(start, joined_on))
    last = date.fromisoformat(end)
    n = 0
    while d <= last:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


def competition_ranks(rows: list) -> list:
    """Standard competition ranking (1, 2, 2, 4) on rows already sorted by total_late_minutes desc."""
    prev_total, prev_rank = None, 0
    for i, r in enumerate(rows, start=1):
        r["rank"] = prev_rank if r["total_late_minutes"] == prev_total else i
        prev_total, prev_rank = r["total_late_minutes"], r["rank"]
    return rows
# --- END PURE HELPERS ------------------------------------------------------------------------------------------ #

load_dotenv()  # does not override real environment variables

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,  # datetimes come back tz-aware (UTC), not naive
    serverSelectionTimeoutMS=3000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]


def ensure_indexes():
    """Idempotent: create_index is a no-op when the index already exists."""
    e, a = db.employees, db.attendance_logs
    e.create_index([("emp_code", ASCENDING)], unique=True, name="uq_emp_code")  # uniqueness, listing sort, lookups
    e.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="dept_emp")  # filtered listing
    e.create_index([("department", ASCENDING), ("joined_on", ASCENDING)], name="dept_joined")  # headcount (R9)
    e.create_index([("joined_on", ASCENDING)], name="joined")  # headcount without a department filter
    a.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_emp_date")  # one record per day
    a.create_index([("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="emp_punch_in")  # punch-out lookup
    a.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="date_emp")  # listing order, month scans
    a.create_index([("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="status_date")


@asynccontextmanager
async def lifespan(_app):
    try:
        ensure_indexes()
    except Exception as exc:  # keep the process up so /health can report 503
        print("index creation failed:", exc)
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


def unprocessable(loc, msg):
    return HTTPException(422, detail=[{"loc": list(loc), "msg": msg, "type": "value_error"}])


def check_date(value: str, name: str) -> str:
    if not valid_date(value):
        raise unprocessable(["query", name], "must be a valid YYYY-MM-DD date")
    return value


def get_employee(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if not emp:
        raise HTTPException(404, "employee not found")
    return emp


# --------------------------------------------------------------------------- #
# Serializers (BSON -> API). _id never leaves; instants become epoch ms.
# --------------------------------------------------------------------------- #
def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"], "name": doc["name"], "email": doc["email"],
        "department": doc["department"], "shift_start": doc["shift_start"], "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"], "created_at": opt_ms(doc.get("created_at")),
    }


def history_out(h: dict) -> dict:
    changes = {}
    for key, ch in (h.get("changes") or {}).items():
        if key in ("punch_in", "punch_out"):
            ch = {"from": opt_ms(ch.get("from")), "to": opt_ms(ch.get("to"))}
        changes[key] = ch
    return {"at": opt_ms(h.get("at")), "by": h.get("by"), "reason": h.get("reason"), "changes": changes}


def record_out(d: dict) -> dict:
    return {
        "emp_code": d["emp_code"], "date": d["date"], "status": d["status"],
        "punch_in": opt_ms(d.get("punch_in")), "punch_out": opt_ms(d.get("punch_out")),
        "work_hours": d.get("work_hours"),
        "late_minutes": d.get("late_minutes") or 0,
        "overtime_minutes": d.get("overtime_minutes") or 0,
        "half_day": bool(d.get("half_day", False)),  # legacy documents omit it
        "history": [history_out(h) for h in (d.get("history") or [])],  # legacy documents omit it
    }


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
EpochMs = Annotated[int, Field(strict=True, ge=MS_MIN, le=MS_MAX)]  # rejects seconds, floats, strings, bools
HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
StatusT = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=HHMM)
    shift_end: str = Field("18:30", pattern=HHMM)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _joined_on(cls, v):
        if not valid_date(v):
            raise ValueError("joined_on must be a valid YYYY-MM-DD date")
        return v

    @model_validator(mode="after")
    def _shifts_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[StatusT] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# System / employees
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        db.command("ping")
    except Exception:
        raise HTTPException(503, "mongodb unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = trunc(datetime.now(UTC))
    try:
        db.employees.insert_one(doc)  # the unique index decides; no check-then-insert
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {"department": department} if department else {}
    total = db.employees.count_documents(q)  # total reflects the applied filter (R10)
    cur = db.employees.find(q, {"_id": 0}).sort("emp_code", ASCENDING).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(d) for d in cur], "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# Attendance
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else trunc(datetime.now(UTC))
    d = attendance_date(ts, emp)
    doc = {
        "emp_code": body.emp_code, "date": d, "status": body.status,
        "punch_in": ts, "punch_out": None,
        **derive(emp, body.status, ts, None, d),
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date) index: exactly one winner under concurrency
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else trunc(datetime.now(UTC))
    # most recent record whose punch_in is at or before the punch-out (works across midnight)
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, sort=[("punch_in", DESCENDING)]
    )
    if not rec:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    pin = trunc(rec["punch_in"])
    if ts <= pin or ts - pin > timedelta(hours=24):
        raise unprocessable(["body", "punched_at"], "must be after punch_in and within 24 hours of it")
    fields = {"punch_out": ts, **derive(emp, rec["status"], pin, ts, rec["date"])}
    # atomic guard: only succeeds while punch_out is still empty, so two simultaneous punch-outs give one 200
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None}, {"$set": fields}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return record_out(updated)


def attendance_filter(emp_code, date_from, date_to, status) -> dict:
    q = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    return q


ATT_SORT = {"date": -1, "emp_code": 1}


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusT] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from:
        check_date(date_from, "date_from")
    if date_to:
        check_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise unprocessable(["query", "date_from"], "date_from must not be after date_to")
    q = attendance_filter(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(q)
    cur = db.attendance_logs.find(q).sort(list(ATT_SORT.items())).skip((page - 1) * page_size).limit(page_size)
    return {"items": [record_out(d) for d in cur], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date_}")
def regularize(emp_code: str, body: RegularizeIn, date_: str = Path(..., alias="date")):
    check_date(date_, "date")
    emp = get_employee(emp_code)
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date_})
    if not rec:
        raise HTTPException(404, "no record for that date")

    status = body.status or rec["status"]
    if status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise unprocessable(["body"], "ABSENT/LEAVE clear punch times; do not send punch_in or punch_out")
        new_in = new_out = None
    else:
        new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else rec.get("punch_in")
        new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else rec.get("punch_out")
        if new_in is None:
            raise unprocessable(["body", "punch_in"], "a presence status requires a punch_in")
        new_in = trunc(new_in)
        new_out = trunc(new_out) if new_out is not None else None
        if attendance_date(new_in, emp) != date_:
            raise unprocessable(["body", "punch_in"], "punch_in must stay on the record's attendance date")
        if new_out is not None and (new_out <= new_in or new_out - new_in > timedelta(hours=24)):
            raise unprocessable(["body", "punch_out"], "punch_out must be after punch_in and within 24 hours")

    new = {"status": status, "punch_in": new_in, "punch_out": new_out, **derive(emp, status, new_in, new_out, date_)}
    old = {
        "status": rec["status"], "punch_in": rec.get("punch_in"), "punch_out": rec.get("punch_out"),
        "work_hours": rec.get("work_hours"), "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0, "half_day": bool(rec.get("half_day", False)),
    }
    changes = {k: {"from": old[k], "to": new[k]} for k in new if old[k] != new[k]}
    if not changes:
        raise unprocessable(["body"], "the request changes nothing")
    entry = {"at": trunc(datetime.now(UTC)), "by": body.regularized_by, "reason": body.reason, "changes": changes}

    # optimistic concurrency: the update only applies if the record is exactly as we read it
    guard = {"_id": rec["_id"], "status": old["status"], "punch_in": old["punch_in"], "punch_out": old["punch_out"]}
    guard["history"] = {"$size": len(rec["history"])} if "history" in rec else {"$exists": False}
    updated = db.attendance_logs.find_one_and_update(
        guard, {"$set": new, "$push": {"history": entry}}, return_document=ReturnDocument.AFTER
    )
    if updated is None:
        raise HTTPException(409, "record was modified concurrently; retry")
    return record_out(updated)


# --------------------------------------------------------------------------- #
# Analytics: pipeline builders (also used, unchanged, by the explain endpoint)
# --------------------------------------------------------------------------- #
def _day_of_week(date_field: str):
    return {"$dayOfWeek": {"$dateFromString": {"dateString": date_field}}}  # 1 = Sunday ... 7 = Saturday


def _present_value(status_f: str, half_f: str):
    """1 for a presence status, 0.5 when half_day, else 0."""
    return {"$cond": [{"$in": [status_f, PRESENCE]},
                      {"$cond": [{"$eq": [{"$ifNull": [half_f, False]}, True]}, 0.5, 1]}, 0]}


def monthly_pipeline(emp_code: str, start: str, end: str) -> list:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lte": end}}},
        {"$group": {
            "_id": None,
            # present_days counts Mon-Fri only (R7); weekend records still count towards late/overtime
            "present_days": {"$sum": {"$cond": [
                {"$in": [_day_of_week("$date"), [2, 3, 4, 5, 6]]}, _present_value("$status", "$half_day"), 0]}},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


def dept_summary_pipeline(start: str, end: str, department: Optional[str]) -> list:
    match = {"joined_on": {"$lte": end}}  # R9: joined on or before the end of the month
    if department:
        match["department"] = department
    has_wh = {"$and": [{"$in": [{"$ifNull": ["$logs.status", "NONE"]}, PRESENCE]}, {"$isNumber": "$logs.work_hours"}]}
    return [
        {"$match": match},
        # start from employees so people with zero logs still produce a row (R9)
        {"$lookup": {"from": "attendance_logs", "localField": "emp_code", "foreignField": "emp_code",
                     "pipeline": [{"$match": {"date": {"$gte": start, "$lte": end}}}], "as": "logs"}},
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {"$addFields": {"dow": {"$cond": [{"$ifNull": ["$logs.date", False]}, _day_of_week("$logs.date"), None]}}},
        {"$group": {
            "_id": "$department",
            "emps": {"$addToSet": "$emp_code"},
            "present_days": {"$sum": {"$cond": [
                {"$in": ["$dow", [2, 3, 4, 5, 6]]}, _present_value("$logs.status", "$logs.half_day"), 0]}},
            # work hours summed as integer hundredths so the later average is exact
            "wh_cents": {"$sum": {"$cond": [has_wh, {"$round": [{"$multiply": ["$logs.work_hours", 100]}, 0]}, 0]}},
            "wh_n": {"$sum": {"$cond": [has_wh, 1, 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$logs.late_minutes", 0]}},
            "leave_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}},
            "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}},
        }},
        {"$project": {"_id": 0, "department": "$_id", "headcount": {"$size": "$emps"}, "present_days": 1,
                      "wh_cents": 1, "wh_n": 1, "late_count": 1, "total_late_minutes": 1,
                      "leave_count": 1, "on_duty_count": 1}},
        {"$sort": {"department": 1}},
    ]


def leaderboard_pipeline(start: str, end: str, department: Optional[str]) -> list:
    p = [
        {"$match": {"date": {"$gte": start, "$lte": end}}},
        {"$group": {
            "_id": "$emp_code",
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
        }},
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # logs whose employee does not exist are dropped here
    ]
    if department:
        p.append({"$match": {"emp.department": department}})
    p += [
        {"$project": {"_id": 0, "emp_code": "$_id", "name": "$emp.name", "department": "$emp.department",
                      "total_late_minutes": 1, "late_count": 1}},
        {"$sort": {"total_late_minutes": -1, "emp_code": 1}},
    ]
    return p


def trend_inputs(department: str):
    emps = list(db.employees.find({"department": department}, {"_id": 0, "emp_code": 1, "joined_on": 1}))
    return [e["emp_code"] for e in emps], [e["joined_on"] for e in emps]


def trend_pipeline(codes: list, joined: list, frm: str, to: str) -> list:
    lo = datetime.combine(date.fromisoformat(frm), time(0), tzinfo=UTC)
    hi = datetime.combine(date.fromisoformat(to), time(0), tzinfo=UTC) + timedelta(days=1)  # upper bound is exclusive
    return [
        {"$match": {"emp_code": {"$in": codes}, "date": {"$gte": frm, "$lte": to}}},
        {"$group": {
            "_id": "$date",
            "present_count": {"$sum": _present_value("$status", "$half_day")},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
        }},
        {"$addFields": {"d": {"$dateFromString": {"dateString": "$_id"}}}},
        {"$densify": {"field": "d", "range": {"step": 1, "unit": "day", "bounds": [lo, hi]}}},  # gap-filling in Mongo
        {"$addFields": {
            "present_count": {"$ifNull": ["$present_count", 0]},
            "late_count": {"$ifNull": ["$late_count", 0]},
            "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$d"}},
            "dow": {"$dayOfWeek": "$d"},
        }},
        {"$addFields": {"headcount": {"$size": {"$filter": {  # R9: joined on or before that day
            "input": {"$literal": joined}, "as": "j", "cond": {"$lte": ["$$j", "$date"]}}}}}},
        {"$sort": {"d": 1}},
        {"$project": {"_id": 0, "date": 1, "dow": 1, "present_count": 1, "late_count": 1, "headcount": 1}},
    ]


MONTH_RE = r"^[0-9]{4}-(0[1-9]|1[0-2])$"


def month_or_422(month: str):
    try:
        return month_bounds(month)
    except ValueError:
        raise unprocessable(["query", "month"], "invalid month")


# --------------------------------------------------------------------------- #
# Analytics endpoints
# --------------------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=MONTH_RE)):
    start, end = month_or_422(month)
    emp = get_employee(emp_code)
    row = next(db.attendance_logs.aggregate(monthly_pipeline(emp_code, start, end)), {})
    wd = working_days(start, end, emp["joined_on"])
    present = float(row.get("present_days", 0))
    pct = None if wd == 0 else half_up(Decimal(str(present)) / Decimal(wd) * 100, 2)
    return {
        "emp_code": emp_code, "month": month, "working_days": wd, "present_days": present,
        "leave_days": int(row.get("leave_days", 0)), "late_count": int(row.get("late_count", 0)),
        "total_late_minutes": int(row.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(row.get("total_overtime_minutes", 0)), "attendance_pct": pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(..., pattern=MONTH_RE), department: Optional[str] = None):
    start, end = month_or_422(month)
    items = []
    for r in db.employees.aggregate(dept_summary_pipeline(start, end, department)):
        avg = None
        if r["wh_n"]:
            avg = half_up(Decimal(int(r["wh_cents"])) / Decimal(r["wh_n"]) / Decimal(100), 2)
        items.append({
            "department": r["department"], "headcount": int(r["headcount"]),
            "present_days": float(r["present_days"]), "avg_work_hours": avg,
            "late_count": int(r["late_count"]), "total_late_minutes": int(r["total_late_minutes"]),
            "leave_count": int(r["leave_count"]), "on_duty_count": int(r["on_duty_count"]),
        })
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=MONTH_RE),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    start, end = month_or_422(month)
    rows = list(db.attendance_logs.aggregate(leaderboard_pipeline(start, end, department)))
    competition_ranks(rows)  # rank is computed from totals, not array position
    items = [
        {"rank": r["rank"], "emp_code": r["emp_code"], "name": r["name"], "department": r["department"],
         "total_late_minutes": int(r["total_late_minutes"]), "late_count": int(r["late_count"])}
        for r in rows if r["rank"] <= limit  # limit applies AFTER ranking, so ties can exceed it
    ]
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_: str = Query(..., alias="from"),
    to: str = Query(...),
):
    check_date(from_, "from")
    check_date(to, "to")
    days = (date.fromisoformat(to) - date.fromisoformat(from_)).days + 1
    if days < 1:
        raise unprocessable(["query", "to"], "to must not be before from")
    if days > 92:
        raise unprocessable(["query", "to"], "range must not exceed 92 days")
    codes, joined = trend_inputs(department)
    if not codes:
        raise HTTPException(404, "unknown department")
    rows = list(db.attendance_logs.aggregate(trend_pipeline(codes, joined, from_, to)))
    items, rates = [], []
    for r in rows:
        working = r["dow"] in (2, 3, 4, 5, 6)
        headcount = int(r["headcount"])
        present = float(r["present_count"])
        rate = half_up(Decimal(str(present)) / Decimal(headcount), 4) if working and headcount > 0 else None
        rates.append(rate)
        window = [x for x in rates[-7:] if x is not None]  # current row + up to 6 before, inside the range only
        avg = half_up(sum(Decimal(str(x)) for x in window) / Decimal(len(window)), 4) if window else None
        items.append({
            "date": r["date"], "is_working_day": working, "headcount": headcount, "present_count": present,
            "late_count": int(r["late_count"]), "attendance_rate": rate, "moving_avg_7d": avg,
        })
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Explain: runs the SAME query/pipeline builders the endpoints use
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=MONTH_RE),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusT] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    def need(**kw):
        for k, v in kw.items():
            if v is None:
                raise unprocessable(["query", k], "required for this endpoint")

    if endpoint == "attendance_list":
        for v, n in ((date_from, "date_from"), (date_to, "date_to")):
            if v:
                check_date(v, n)
        if date_from and date_to and date_from > date_to:
            raise unprocessable(["query", "date_from"], "date_from must not be after date_to")
        cmd = {"find": "attendance_logs", "filter": attendance_filter(emp_code, date_from, date_to, status),
               "sort": ATT_SORT, "skip": (page - 1) * page_size, "limit": page_size}
        coll = "attendance_logs"
    elif endpoint == "employee_monthly":
        need(emp_code=emp_code, month=month)
        s, e = month_or_422(month)
        cmd = {"aggregate": "attendance_logs", "pipeline": monthly_pipeline(emp_code, s, e), "cursor": {}}
        coll = "attendance_logs"
    elif endpoint == "department_summary":
        need(month=month)
        s, e = month_or_422(month)
        cmd = {"aggregate": "employees", "pipeline": dept_summary_pipeline(s, e, department), "cursor": {}}
        coll = "employees"
    elif endpoint == "late_leaderboard":
        need(month=month)
        s, e = month_or_422(month)
        cmd = {"aggregate": "attendance_logs", "pipeline": leaderboard_pipeline(s, e, department), "cursor": {}}
        coll = "attendance_logs"
    else:  # department_trend
        need(department=department, **{"from": from_}, to=to)
        check_date(from_, "from")
        check_date(to, "to")
        if (date.fromisoformat(to) - date.fromisoformat(from_)).days + 1 < 1:
            raise unprocessable(["query", "to"], "to must not be before from")
        codes, joined = trend_inputs(department)
        cmd = {"aggregate": "attendance_logs", "pipeline": trend_pipeline(codes, joined, from_, to), "cursor": {}}
        coll = "attendance_logs"

    raw = db.command("explain", cmd, verbosity="executionStats")
    return {"endpoint": endpoint, "collection": coll,
            "explain": json.loads(json_util.dumps(raw, json_options=json_util.RELAXED_JSON_OPTIONS))}
