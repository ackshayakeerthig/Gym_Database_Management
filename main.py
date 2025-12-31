import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from passlib.context import CryptContext
from datetime import datetime, timedelta
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/GYM_MANAGEMENT")
# --- CONFIGURATION ---
DB_CONFIG = {
    "host": "localhost",
    "database": "GYM_MANAGEMENT",
    "user": "postgres",
    "password": "postgres",
    "port": "5432"
}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = FastAPI(title="GymTech Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE UTILS ---
def get_db():
    # conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

# --- AUTH MODELS ---
class LoginRequest(BaseModel):
    username: str
    password: str

class ProfileUpdate(BaseModel):
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None

# --- AUTH LOGIC (Auto-detect Role) ---
@app.post("/login")
def login(req: LoginRequest):
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    
    # 1. Check Members Table
    cur.execute("SELECT member_id as id, full_name as name, password_hash, 'Member' as role FROM Members WHERE username = %s", (req.username,))
    user = cur.fetchone()
    
    # 2. Check Employees Table if not found in Members
    if not user:
        cur.execute("SELECT employee_id as id, name, password_hash, role FROM Employees WHERE username = %s", (req.username,))
        user = cur.fetchone()
        if user and user['role'] == 'Manager':
            user['role'] = 'Manager'
        elif user:
            user['role'] = 'Employee'

    if not user or not pwd_context.verify(req.password, user['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    return {
        "id": user['id'],
        "name": user['name'],
        "role": user['role'],
        "status": "success"
    }

# --- MEMBER MODULE ---

@app.get("/member/{id}/profile")
def get_member_profile(id: int):
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT m.*, e.plan_name as current_plan 
        FROM Members m 
        LEFT JOIN Membership_Enrollments me ON m.member_id = me.member_id 
        LEFT JOIN Subscription_Plans e ON me.plan_id = e.plan_id 
        WHERE m.member_id = %s 
        ORDER BY me.end_date DESC LIMIT 1
    """, (id,))
    return cur.fetchone()

@app.put("/member/{id}/profile")
def update_member_profile(id: int, data: ProfileUpdate):
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE Members 
            SET phone = COALESCE(%s, phone), 
                email = COALESCE(%s, email), 
                address = COALESCE(%s, address) 
            WHERE member_id = %s
        """, (data.phone, data.email, data.address, id))
        conn.commit()
    except psycopg2.errors.CheckViolation:
        conn.rollback()
        raise HTTPException(status_code=400, detail="Invalid phone number. Must be a 10-digit Indian mobile number.")
    finally:
        cur.close()
        conn.close()
    return {"message": "Profile updated"}


@app.get("/member/{id}/calendar")
def get_workout_calendar(id: int):
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT recorded_at, activity_type, details FROM Member_Activity_Docs WHERE member_id = %s ORDER BY recorded_at DESC", (id,))
    return cur.fetchall()

@app.get("/equipment/status")
def get_gym_health():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT status, COUNT(*) FROM Equipment_Assets GROUP BY status")
    stats = cur.fetchall()
    cur.execute("SELECT asset_name, status FROM Equipment_Assets WHERE status != 'Functional'")
    broken = cur.fetchall()
    return {"summary": stats, "details": broken}

# --- EMPLOYEE MODULE ---

@app.get("/employee/colleagues")
def get_colleagues():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT name, role, email, phone FROM Employees")
    return cur.fetchall()

@app.get("/classes/all")
def get_all_classes():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, e.name as trainer_name, 
        (SELECT COUNT(*) FROM Class_Bookings WHERE schedule_id = s.schedule_id) as booked_count
        FROM Class_Schedules s JOIN Employees e ON s.trainer_id = e.employee_id
    """)
    return cur.fetchall()

@app.patch("/attendance/{booking_id}")
def mark_attendance(booking_id: int, attended: bool):
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("UPDATE Class_Bookings SET attended = %s WHERE booking_id = %s", (attended, booking_id))
    conn.commit()
    return {"message": "Attendance updated"}

@app.get("/inventory/all")
def get_inventory():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT *, (current_stock < 10) as low_stock FROM Inventory_Catalog")
    return cur.fetchall()

# --- MANAGER MODULE ---

@app.get("/manager/analytics")
def get_financials():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT SUM(final_price_paid) FROM Membership_Enrollments")
    revenue = cur.fetchone()['sum'] or 0
    cur.execute("SELECT SUM(total_bulk_cost) FROM Wholesale_Orders")
    expenses = cur.fetchone()['sum'] or 0
    return {"total_revenue": revenue, "total_expenses": expenses, "net_profit": revenue - expenses}

@app.get("/manager/staff")
def get_staff_management():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT e1.*, e2.name as manager_name 
        FROM Employees e1 
        LEFT JOIN Employees e2 ON e1.reports_to = e2.employee_id
    """)
    return cur.fetchall()

# --- AGENT CHAT (Dummy API for now) ---
@app.post("/api/chat")
def chat_endpoint(message: str, user_id: int, role: str):
    # This will be replaced by your FAISS/Agent logic later
    return {
        "answer": f"Agent received your message: '{message}'. Entity resolution for role {role} is in progress.",
        "status": "simulated"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)