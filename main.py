import bcrypt
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

# This forces passlib to use the 'bcrypt' package instead of its broken internal logic
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto"
)
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
# ---other models----
class ClassCreate(BaseModel):
    class_name: str
    trainer_id: int
    start_time: datetime
    capacity: int

class StatusUpdate(BaseModel):
    status: str

class StockUpdate(BaseModel):
    current_stock: int

class StaffUpdate(BaseModel):
    salary: Optional[float] = None
    reports_to: Optional[int] = None

class BookingCreate(BaseModel):
    member_id: int
    schedule_id: int

class PurchaseRequest(BaseModel):
    member_id: int
    item_id: int
    quantity: int
# --- AUTH LOGIC (Auto-detect Role) ---
@app.post("/login")
def login(req: LoginRequest):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
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
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
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
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
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
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT recorded_at, activity_type, details FROM Member_Activity_Docs WHERE member_id = %s ORDER BY recorded_at DESC", (id,))
    return cur.fetchall()

@app.get("/equipment/status")
def get_gym_health():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT status, COUNT(*) FROM Equipment_Assets GROUP BY status")
    stats = cur.fetchall()
    cur.execute("SELECT asset_name, status FROM Equipment_Assets WHERE status != 'Functional'")
    broken = cur.fetchall()
    return {"summary": stats, "details": broken}

@app.get("/member/{id}/subscriptions")
def get_member_subscriptions(id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT me.*, sp.plan_name, sp.duration_months 
        FROM Membership_Enrollments me 
        JOIN Subscription_Plans sp ON me.plan_id = sp.plan_id 
        WHERE me.member_id = %s ORDER BY me.start_date DESC
    """, (id,))
    return cur.fetchall()

@app.get("/plans")
def get_all_plans():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM Subscription_Plans")
    return cur.fetchall()

@app.get("/member/{id}/bookings")
def get_member_bookings(id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT cb.*, cs.class_name, cs.start_time, e.name as trainer_name 
        FROM Class_Bookings cb 
        JOIN Class_Schedules cs ON cb.schedule_id = cs.schedule_id 
        JOIN Employees e ON cs.trainer_id = e.employee_id 
        WHERE cb.member_id = %s
    """, (id,))
    return cur.fetchall()
    
@app.post("/bookings")
def create_booking(req: BookingCreate):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    try:
        # We don't need manual checks here because our SQL triggers:
        # 1. trg_cap: Checks for class capacity
        # 2. trg_verify: Checks for Active Paid Membership
        # 3. unique_member_class_booking: Prevents double booking
        
        cur.execute("""
            INSERT INTO Class_Bookings (member_id, schedule_id, booking_date, attended) 
            VALUES (%s, %s, CURRENT_DATE, False)
        """, (req.member_id, req.schedule_id))
        
        conn.commit()
        return {"message": "Booking successful", "status": "success"}

    except psycopg2.Error as e:
        conn.rollback()
        # This catches the RAISE EXCEPTION messages from your SQL triggers
        # e.g., "Class is full!" or "No active membership on this date"
        error_msg = str(e.pgerror).split("CONTEXT:")[0] if hasattr(e, 'pgerror') else str(e)
        raise HTTPException(status_code=400, detail=error_msg)
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        cur.close()
        conn.close()


@app.get("/classes/available")
def get_available_classes():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    # Shows classes in the future that aren't full
    cur.execute("""
        SELECT s.*, e.name as trainer_name, 
        (s.capacity - (SELECT COUNT(*) FROM Class_Bookings WHERE schedule_id = s.schedule_id)) as spots_left
        FROM Class_Schedules s JOIN Employees e ON s.trainer_id = e.employee_id
        WHERE s.start_time > CURRENT_TIMESTAMP
    """)
    return cur.fetchall()

@app.delete("/bookings/{booking_id}")
def cancel_booking(booking_id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("DELETE FROM Class_Bookings WHERE booking_id = %s", (booking_id,))
    conn.commit()
    return {"message": "Booking cancelled"}

@app.get("/member/{id}/purchases")
def get_member_purchases(id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT p.*, i.item_name 
        FROM Point_Of_Sale p 
        JOIN Inventory_Catalog i ON p.item_id = i.item_id 
        WHERE p.member_id = %s ORDER BY p.sale_timestamp DESC
    """, (id,))
    return cur.fetchall()

@app.post("/member/purchase")
def member_purchase(req: PurchaseRequest):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    try:
        # Calculate total price based on catalog price
        cur.execute("SELECT unit_selling_price FROM Inventory_Catalog WHERE item_id = %s", (req.item_id,))
        item = cur.fetchone()
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        
        total_amount = item['unit_selling_price'] * req.quantity
        
        # Insert into Point_Of_Sale. 
        # THE SUPABASE TRIGGER WILL AUTOMATICALLY REDUCE THE STOCK NOW.
        cur.execute("""
            INSERT INTO Point_Of_Sale (member_id, item_id, quantity, total_amount, sale_timestamp)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        """, (req.member_id, req.item_id, req.quantity, total_amount))
        
        conn.commit()
        return {"status": "success", "message": "Transaction recorded"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cur.close()
        conn.close()

# --- EMPLOYEE MODULE ---

@app.get("/employee/colleagues")
def get_colleagues():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT name, role, email, phone FROM Employees")
    return cur.fetchall()

@app.get("/classes/all")
def get_all_classes():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, e.name as trainer_name, 
        (SELECT COUNT(*) FROM Class_Bookings WHERE schedule_id = s.schedule_id) as booked_count
        FROM Class_Schedules s JOIN Employees e ON s.trainer_id = e.employee_id
    """)
    return cur.fetchall()
    
@app.get("/employee/equipment")
def get_all_equipment_employee():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM Equipment_Assets ORDER BY asset_name ASC")
    return cur.fetchall()

@app.get("/employee/{trainer_id}/classes")
def get_trainer_specific_classes(trainer_id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    try:
        # Fetches only classes for this trainer and counts bookings for each
        cur.execute("""
            SELECT s.*, 
            (SELECT COUNT(*)::int FROM Class_Bookings WHERE schedule_id = s.schedule_id) as booked_count
            FROM Class_Schedules s 
            WHERE s.trainer_id = %s
            ORDER BY s.start_time DESC
        """, (trainer_id,))
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()
        
@app.get("/employee/{trainer_id}/class-stats")
def get_trainer_class_stats(trainer_id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    # Count Past, Today, and Upcoming
    cur.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE start_time < CURRENT_DATE) as past,
            COUNT(*) FILTER (WHERE start_time::date = CURRENT_DATE) as today,
            COUNT(*) FILTER (WHERE start_time > CURRENT_DATE + interval '1 day') as upcoming
        FROM Class_Schedules 
        WHERE trainer_id = %s
    """, (trainer_id,))
    return cur.fetchone()
    
@app.patch("/attendance/{booking_id}")
def mark_attendance(booking_id: int, attended: bool):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("UPDATE Class_Bookings SET attended = %s WHERE booking_id = %s", (attended, booking_id))
    conn.commit()
    return {"message": "Attendance updated"}

@app.get("/inventory/all")
def get_inventory():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT *, (current_stock < 10) as low_stock FROM Inventory_Catalog")
    return cur.fetchall()
@app.get("/employee/{id}/profile")
def get_employee_profile(id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT employee_id, name, username, email, phone, address, role, salary FROM Employees WHERE employee_id = %s", (id,))
    return cur.fetchone()

@app.get("/classes/{schedule_id}/attendees")
def get_class_attendees(schedule_id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT m.full_name, m.email, m.phone, cb.attended, cb.booking_id 
        FROM Class_Bookings cb 
        JOIN Members m ON cb.member_id = m.member_id 
        WHERE cb.schedule_id = %s
    """, (schedule_id,))
    return cur.fetchall()

@app.post("/classes")
def create_class(data: ClassCreate):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("INSERT INTO Class_Schedules (class_name, trainer_id, start_time, capacity) VALUES (%s, %s, %s, %s)",
                (data.class_name, data.trainer_id, data.start_time, data.capacity))
    conn.commit()
    return {"message": "Class scheduled"}

@app.patch("/equipment/{id}")
def update_equipment_status(id: int, data: StatusUpdate):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    try:
        cur.execute("UPDATE Equipment_Assets SET status = %s WHERE asset_id = %s", (data.status, id))
        conn.commit()
        # Fetch updated record to return to frontend
        cur.execute("SELECT * FROM Equipment_Assets WHERE asset_id = %s", (id,))
        return cur.fetchone()
    finally:
        cur.close()
        conn.close()

@app.patch("/inventory/{id}")
def update_inventory_stock(id: int, data: StockUpdate):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    try:
        cur.execute("UPDATE Inventory_Catalog SET current_stock = %s WHERE item_id = %s", (data.current_stock, id))
        conn.commit()
        # Fetch updated record
        cur.execute("SELECT *, (current_stock < 10) as low_stock FROM Inventory_Catalog WHERE item_id = %s", (id,))
        return cur.fetchone()
    finally:
        cur.close()
        conn.close()

@app.get("/suppliers")
def get_all_suppliers():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM Suppliers")
    return cur.fetchall()

@app.get("/members/all")
def get_all_members_admin():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM Members")
    return cur.fetchall()

@app.get("/maintenance/logs")
def get_maintenance_logs():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT ml.*, ea.asset_name, e.name as staff_name 
        FROM Maintenance_Logs ml 
        JOIN Equipment_Assets ea ON ml.asset_id = ea.asset_id 
        JOIN Employees e ON ml.performed_by = e.employee_id
    """)
    return cur.fetchall()
# --- MANAGER MODULE ---

@app.get("/manager/analytics")
def get_financials():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT SUM(final_price_paid) FROM Membership_Enrollments")
    revenue = cur.fetchone()['sum'] or 0
    cur.execute("SELECT SUM(total_bulk_cost) FROM Wholesale_Orders")
    expenses = cur.fetchone()['sum'] or 0
    return {"total_revenue": revenue, "total_expenses": expenses, "net_profit": revenue - expenses}

@app.get("/manager/staff")
def get_staff_management():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        SELECT e1.*, e2.name as manager_name 
        FROM Employees e1 
        LEFT JOIN Employees e2 ON e1.reports_to = e2.employee_id
    """)
    return cur.fetchall()
@app.patch("/manager/staff/{id}")
def update_staff_admin(id: int, data: StaffUpdate):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("""
        UPDATE Employees 
        SET salary = COALESCE(%s, salary), 
            reports_to = COALESCE(%s, reports_to) 
        WHERE employee_id = %s
    """, (data.salary, data.reports_to, id))
    conn.commit()
    return {"message": "Staff record updated"}

@app.get("/manager/trainer-performance/{id}")
def get_trainer_stats(id: int):
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    # Total classes conducted
    cur.execute("SELECT COUNT(*) FROM Class_Schedules WHERE trainer_id = %s AND start_time < CURRENT_TIMESTAMP", (id,))
    classes_done = cur.fetchone()['count']
    # Avg booking rate
    cur.execute("""
        SELECT AVG(booked) FROM (
            SELECT COUNT(cb.booking_id) as booked 
            FROM Class_Schedules cs 
            LEFT JOIN Class_Bookings cb ON cs.schedule_id = cb.schedule_id 
            WHERE cs.trainer_id = %s GROUP BY cs.schedule_id
        ) as sub
    """, (id,))
    avg_bookings = cur.fetchone()['avg'] or 0
    return {"classes_conducted": classes_done, "average_bookings": float(avg_bookings)}


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








