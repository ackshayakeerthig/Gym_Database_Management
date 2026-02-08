import bcrypt
import os, json, operator, psycopg2, numpy as np
from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import  List, Optional, Annotated, TypedDict, Union, Literal
import psycopg2
from psycopg2.extras import RealDictCursor
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import requests
# from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Fetch variables using os.getenv
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256") # Default to HS256 if not found
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


# Debug check (Optional: remove this after testing)
if not DATABASE_URL:
    print("CRITICAL ERROR: DATABASE_URL not found in .env file")


def get_connection():
    # We use RealDictCursor so we can access columns by name like res['display_name']
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    with conn.cursor() as cur:
        # This tells Postgres to look in 'public' and then 'extensions' 
        # for every single query in this session
        cur.execute("SET search_path TO public, extensions, pg_catalog;")
        conn.commit()
        
    return conn


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "VD3kjjhGJdZLj0oZhv29tKYmMG7AgVoJ")

app = FastAPI(title="GymTech Pro AI Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=MISTRAL_API_KEY)
memory = MemorySaver()

# --- DATABASE UTILS ---
def get_db():
    # conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()

security = HTTPBearer()

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

class ChatRequest(BaseModel):
    message: str
    user_id: int
    role: str
    session_id: str  # Added session_id to keep the conversation "saved"

class ActivityLogCreate(BaseModel):
    member_id: int
    activity_type: Literal["Workout", "Health_Check"]
    details: dict  # This will store the flexible form fields

class MemberCreate(BaseModel):
    full_name: str
    username: str
    password: str
    email: str
    phone: str
    address: str

class MaintenanceLogCreate(BaseModel):
    asset_id: int
    maintenance_date: datetime
    repair_cost: float
    notes: Optional[str] = ""

class User(BaseModel):
    id: int
    name: str
    role: str

import time
import requests

def get_huggingface_embedding(text: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": [text]} # Wrap in list as required by /pipeline/
    
    # Try up to 3 times in case the model is "sleeping"
    for attempt in range(3):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                # Based on your Colab result, we take the first vector
                return data[0] 
            
            elif response.status_code == 503:
                # Model is loading. Wait for 'estimated_time' or default 5s
                est_time = response.json().get("estimated_time", 5)
                print(f"HF Model loading... waiting {est_time}s (Attempt {attempt+1}/3)")
                time.sleep(min(est_time, 10)) # Don't wait more than 10s per retry
                
            else:
                print(f"HF Error {response.status_code}: {response.text}")
                break
        except Exception as e:
            print(f"Request failed: {e}")
            
    return None

def resolve_entity_via_vector(query):
    # q_emb = model.encode(query).tolist()
    q_emb = get_huggingface_embedding(query)
    conn = get_connection()
    cur = conn.cursor()
    try:
        # cur.execute("SELECT current_database(), current_user, inet_server_addr();")
        # info = cur.fetchone()
        # print(f"--- PYTHON CONNECTED TO ---")
        # print(f"DB Name: {info['current_database']}")
        # print(f"User: {info['current_user']}")
        # print(f"Server IP: {info['inet_server_addr']}")
        
        # # 2. LIST ALL SCHEMAS PYTHON CAN SEE
        # cur.execute("SELECT schema_name FROM information_schema.schemata")
        # schemas = [s['schema_name'] for s in cur.fetchall()]
        # print(f"Schemas available to Python: {schemas}")

        # Force the session to recognize 'public' and 'extensions' schemas
        cur.execute("SET search_path TO public, extensions, pg_catalog;")
        
        # Use the fully qualified "public"."global_search_vectors" name
        cur.execute('''
            SELECT original_id, display_name, entity_type, (embedding <=> %s::vector) AS dist 
            FROM "public"."global_search_vectors" 
            ORDER BY dist ASC LIMIT 1
        ''', (q_emb,))
        
        match = cur.fetchone()
        return match if (match and match['dist'] < 0.6) else None
    except Exception as e:
        print(f"CRITICAL VEC ERROR: {str(e)}")
        return None
    finally:
        cur.close()
        conn.close()

# --- 2. THE TOOLS (PROPERLY DEFINED) ---

@tool
def get_user_profile(target_name: Optional[str] = None):
    """Fetches full profile (Name, Phone, Email, Plan). If target_name is empty, uses current user."""
    return "Executing Profile Lookup..."

@tool
def tool_member_records(data_type: Literal["subscriptions", "bookings", "purchases"], target_name: Optional[str] = None):
    """Fetches payment history, class bookings, or shop purchases."""
    return "Executing Records Lookup..."

@tool
def tool_nosql_logs(log_type: Literal["workout_calendar", "health_checks"], target_name: Optional[str] = None):
    """Fetches NoSQL JSONB data: workout history (check-in, duration) or health stats (BMI, weight)."""
    return "Executing NoSQL Retrieval..."

@tool
def tool_gym_ops(query: Literal["equipment_health", "inventory_alerts", "revenue", "plans", "upcoming_classes", "all_classes", "trainer_schedule"], target_name: Optional[str] = None):
    """
    Public gym info, operational data, or schedules.
    'upcoming_classes': General list of classes with spots left.
    'all_classes': Detailed list of all schedules (Staff/Manager only).
    'trainer_schedule': Classes conducted by a specific trainer name (Staff/Manager only).
    """
    return "Executing Operations Tool..."

@tool
def record_workout(exercise: str, weight: str):
    """Logs a new workout entry for the current user."""
    return "Recording Activity..."

ALL_TOOLS_LIST = [get_user_profile, tool_member_records, tool_nosql_logs, tool_gym_ops, record_workout]

# --- 3. LANGGRAPH LOGIC ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_context: dict

def assistant_node(state: AgentState):
    user = state['user_context']
    role = user['role']
    
    # Tool permissions logic
    if role in ["Manager", "Employee"]:
        allowed_tools = ALL_TOOLS_LIST
    else: # Member
        allowed_tools = [get_user_profile, tool_member_records, tool_nosql_logs, record_workout]

    llm_with_tools = llm.bind_tools(allowed_tools)
    
    # STRICT SYSTEM PROMPT
    sys_msg = SystemMessage(content=f"""
    You are the GymTech Pro Agent. UserID: {user['id']}, Role: {role}.
    
    CRITICAL RULES:
    1. NEVER make up numbers for revenue, profit, or equipment status. 
    2. If the user asks about "performance", "revenue", "subscription plans" , "membership types" , "broken machines", "classes" or "inventory alerts", you MUST call 'tool_gym_ops'.
    3. If the tool returns a result, describe the data EXACTLY as provided.
    4. If the role is 'Member', you cannot access 'revenue' or other members' data.
    """)
    
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
def execution_node(state: AgentState):
    user = state['user_context']
    role = user['role']
    tool_outputs = []
    
    for call in state["messages"][-1].tool_calls:
        name, args = call["name"], call["args"]
        tid = user['id']
        t_name = args.get("target_name")
        res = "No data found."
        
        # 1. Identity Resolution
        actual_name = "Self"
        if t_name:
            match = resolve_entity_via_vector(t_name)
            if match:
                tid, actual_name = match['original_id'], match['display_name']
                print(f"[AGENT: IDENTITY] Resolved '{t_name}' to {actual_name} (ID:{tid})")
            else:
                tool_outputs.append(ToolMessage(tool_call_id=call["id"], content=f"Could not find '{t_name}'."))
                continue

        # 2. Execution
        conn = get_connection(); cur = conn.cursor()
        try:
            if name == "get_user_profile":
                cur.execute("SELECT full_name, email, phone, address FROM members WHERE member_id = %s", (tid,))
                res = {"Target": actual_name, "Profile": cur.fetchone()}

            elif name == "tool_member_records":
                tbl = {"subscriptions": "membership_enrollments", "bookings": "class_bookings", "purchases": "point_of_sale"}[args['data_type']]
                cur.execute(f"SELECT * FROM {tbl} WHERE member_id = %s", (tid,))
                res = {"Target": actual_name, "Records": cur.fetchall()}

            elif name == "tool_gym_ops":
                q = args['query']
                if q == "revenue":
                    cur.execute("SELECT SUM(final_price_paid) as r FROM membership_enrollments")
                    rev = cur.fetchone()['r'] or 0
                    cur.execute("SELECT SUM(total_bulk_cost) as e FROM wholesale_orders")
                    exp = cur.fetchone()['e'] or 0
                    res = {"Report": "Financial Performance", "Total_Revenue": float(rev), "Total_Expenses": float(exp), "Net_Profit": float(rev - exp)}
                elif q == "equipment_health":
                    cur.execute("SELECT status, COUNT(*) FROM equipment_assets GROUP BY status")
                    res = {"Equipment_Status": cur.fetchall()}
                elif q == "inventory_alerts":
                    cur.execute("SELECT item_name, current_stock FROM inventory_catalog WHERE current_stock < 20")
                    res = {"Alerts": cur.fetchall()}
                elif q == "plans":
                    cur.execute("SELECT plan_name, duration_months, base_price, description FROM subscription_plans")
                    res = {"Available_Plans": cur.fetchall()}
                elif q == "upcoming_classes":
                    # Fetches classes in the future with availability
                    cur.execute("""
                        SELECT s.class_name, s.start_time, e.name as trainer, 
                        (s.capacity - (SELECT COUNT(*) FROM Class_Bookings WHERE schedule_id = s.schedule_id)) as spots_left 
                        FROM Class_Schedules s 
                        JOIN Employees e ON s.trainer_id = e.employee_id 
                        WHERE s.start_time > CURRENT_TIMESTAMP 
                        ORDER BY s.start_time ASC
                    """)
                    res = {"Upcoming_Classes": cur.fetchall()}

                elif q == "all_classes":
                    if role not in ["Employee", "Manager"]:
                        res = "🚫 Access Denied: Only staff can view the full schedule."
                    else:
                        cur.execute("""
                            SELECT s.*, e.name as trainer, 
                            (SELECT COUNT(*) FROM class_bookings WHERE schedule_id = s.schedule_id) as current_bookings 
                            FROM Class_Schedules s 
                            JOIN Employees e ON s.trainer_id = e.employee_id 
                            ORDER BY s.start_time DESC
                        """)
                        res = {"All_Schedules": cur.fetchall()}

                elif q == "trainer_schedule":
                    if role not in ["Employee", "Manager"]:
                        res = "🚫 Access Denied."
                    else:
                        # Resolve trainer name if provided, otherwise default to self
                        search_name = args.get("target_name")
                        if search_name:
                            match = resolve_entity_via_vector(search_name)
                            if match and match['entity_type'] == 'Employee':
                                t_id, actual_t_name = match['original_id'], match['display_name']
                            else:
                                res = f"Trainer '{search_name}' not found."
                                break
                        else:
                            t_id, actual_t_name = user['id'], "Yourself"

                        cur.execute("""
                            SELECT class_name, start_time, capacity, 
                            (SELECT COUNT(*) FROM Class_Bookings WHERE schedule_id = s.schedule_id) as attendees
                            FROM Class_Schedules s 
                            WHERE trainer_id = %s ORDER BY start_time DESC
                        """, (t_id,))
                        res = {"Trainer": actual_t_name, "Schedule": cur.fetchall()}
            elif name == "tool_nosql_logs":
                atype = 'Workout' if args['log_type'] == 'workout_calendar' else 'Health_Check'
                cur.execute("SELECT details, recorded_at FROM member_activity_docs WHERE member_id=%s AND activity_type=%s", (tid, atype))
                res = {"Target": actual_name, "NoSQL_Logs": cur.fetchall()}

        except Exception as e:
            res = f"Error: {str(e)}"
        
        cur.close(); conn.close()
        tool_outputs.append(ToolMessage(tool_call_id=call["id"], content=json.dumps(res, default=str)))
        
    return {"messages": tool_outputs}

# --- 4. COMPILE & CHAT ---
builder = StateGraph(AgentState)
builder.add_node("agent", assistant_node); builder.add_node("tools", execution_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", lambda x: "tools" if x["messages"][-1].tool_calls else END)
builder.add_edge("tools", "agent")
gym_app = builder.compile(checkpointer=memory)

history = {}
def chat(uid, sid, msg, role="Member"):
    k = f"{uid}:{sid}"
    if k not in history: history[k] = []
    
    # Add new message to history
    current_history = history[k] + [HumanMessage(content=msg)]
    
    state = {"messages": current_history, "user_context": {'id': uid, 'role': role}}
    
    # Run graph
    out = gym_app.invoke(state)
    
    # Save the FULL message chain back to history
    history[k] = out["messages"]
    
    # Return the final response string
    return out["messages"][-1].content


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # Pass the global SECRET_KEY and ALGORITHM
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        uid = payload.get("sub")
        role = payload.get("role")
        name = payload.get("name")
        
        if uid is None:
            print("AUTH ERROR: Token is missing 'sub' field")
            raise HTTPException(status_code=401, detail="Invalid token")
            
        return User(id=uid, name=name, role=role)
        
    except jwt.ExpiredSignatureError:
        print("AUTH ERROR: Token has expired")
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        print(f"AUTH ERROR: Invalid Token - {str(e)}") # This prints the EXACT reason
        raise HTTPException(status_code=401, detail="Invalid token")

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
    
    token = create_access_token(data={
        "sub": str(user['id']), 
        "role": user['role'], 
        "name": user['name']
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user['id'],
            "name": user['name'],
            "role": user['role']
        }
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

# main.py
@app.post("/api/classes")
async def create_class(req: ClassCreate, current_user: User = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        # We add ::timestamp to ensure Postgres treats the string correctly
        cur.execute("""
            INSERT INTO class_schedules (class_name, start_time, trainer_id, capacity)
            VALUES (%s, %s::timestamp, %s, %s)
            RETURNING schedule_id
        """, (req.class_name, req.start_time, current_user.id, req.capacity))
        
        new_id = cur.fetchone()['schedule_id']
        conn.commit()
        return {"status": "success", "schedule_id": new_id}
    except Exception as e:
        conn.rollback()
        print(f"Database Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create class: {str(e)}")
    finally:
        cur.close(); conn.close()


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


@app.post("/maintenance/logs")
def create_maintenance_log(req: MaintenanceLogCreate, current_user: User = Depends(get_current_user)):
    if current_user.role not in ["Employee", "Manager"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
        
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO maintenance_logs (asset_id, maintenance_date, performed_by, repair_cost, notes)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING log_id
        """, (req.asset_id, req.maintenance_date, current_user.id, req.repair_cost, req.notes))
        
        conn.commit()
        return {"status": "success", "message": "Maintenance log added"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cur.close(); conn.close()


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
    
@app.post("/employee/add-member")
def add_member(req: MemberCreate, current_user: User = Depends(get_current_user)):
    # Security: Ensure only staff can add members
    if current_user.role not in ["Employee", "Manager"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
        
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Hash the default password
        pwd_hash = pwd_context.hash(req.password)
        
        cur.execute("""
            INSERT INTO Members (full_name, username, password_hash, email, phone, address)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING member_id
        """, (req.full_name, req.username, pwd_hash, req.email, req.phone, req.address))
        
        new_id = cur.fetchone()['member_id']
        conn.commit()
        return {"status": "success", "member_id": new_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cur.close(); conn.close()

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

@app.post("/employee/log-activity")
def log_member_activity(req: ActivityLogCreate, current_user: User = Depends(get_current_user)):
    # Ensure only Staff can write to the NoSQL Log Store (DS3)
    if current_user.role not in ["Employee", "Manager"]:
        raise HTTPException(status_code=403, detail="Members cannot log their own activity docs")
    
    conn = get_connection(); cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO member_activity_docs (member_id, activity_type, details, recorded_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        """, (req.member_id, req.activity_type, json.dumps(req.details)))
        conn.commit()
        return {"status": "success", "message": f"{req.activity_type} log recorded"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        cur.close(); conn.close()

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
    cur.execute("SELECT * FROM suppliers ORDER BY company_name ASC")
    return cur.fetchall()

@app.get("/members/all")
def get_all_members_admin():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM Members")
    return cur.fetchall()

@app.get("/maintenance/logs")
def get_maintenance_logs():
    conn = get_connection()
    cur = conn.cursor()
    # Joining tables to get Asset Names and Staff Names
    cur.execute("""
        SELECT ml.log_id, ml.maintenance_date, ml.repair_cost, ml.notes,
               ea.asset_name, e.name as staff_name 
        FROM maintenance_logs ml 
        JOIN equipment_assets ea ON ml.asset_id = ea.asset_id 
        JOIN employees e ON ml.performed_by = e.employee_id
        ORDER BY ml.maintenance_date DESC
    """)
    return cur.fetchall()
# --- MANAGER MODULE ---

@app.get("/manager/analytics")
def get_financials(current_user: User = Depends(get_current_user)):
    # Block non-managers from seeing revenue/expenses
    if current_user.role != "Manager":
        raise HTTPException(status_code=403, detail="Managerial access required for financial data")
    
    conn = get_connection(); cur = conn.cursor()
    cur.execute("SELECT SUM(final_price_paid) as r FROM Membership_Enrollments")
    revenue = cur.fetchone()['r'] or 0
    cur.execute("SELECT SUM(total_bulk_cost) as e FROM Wholesale_Orders")
    expenses = cur.fetchone()['e'] or 0
    cur.close(); conn.close()
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

# main.py or routes/employees.py

@app.get("/api/employees")
async def get_all_employees(current_user: User = Depends(get_current_user) ):
    if current_user.role != "Manager":
        raise HTTPException(status_code=403, detail="Managerial access only")
        
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT employee_id, name, email, role, salary FROM employees")
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

@app.patch("/api/employees/{emp_id}")
async def update_employee(emp_id: int, data: dict ,current_user: User = Depends(get_current_user)   ):
    if current_user.role != "Manager":
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE employees SET salary = %s, role = %s WHERE employee_id = %s",
        (data['salary'], data['position'], emp_id)
    )
    conn.commit()
    cur.close(); conn.close()
    return {"message": "Success"}

# --- AGENT CHAT (Dummy API for now) ---
# @app.post("/api/chat")
# def agent_chat(message: str, current_user_id: int, current_role: str):
#     conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
#     cur = conn.cursor()
    
#     try:
#         # 1. VECTOR SEARCH
#         # Convert user message into a vector (list of floats)
#         query_embedding = model.encode(message).tolist()
        
#         # Search the Global_Search_Vectors table using Cosine Distance (<=>)
#         cur.execute("""
#             SELECT entity_type, original_id, display_name, 
#                    (embedding <=> %s::vector) as distance
#             FROM global_search_vectors
#             ORDER BY distance ASC LIMIT 1
#         """, (query_embedding,))
        
#         match = cur.fetchone()
        
#         # 2. THE ROUTER
#         # 0.6 is a common threshold; higher means the match is too "far" away
#         if not match or match['distance'] > 0.6:
#             return {"answer": "I couldn't find any specific information about that. Could you be more specific?"}

#         e_type = match['entity_type']
#         e_id = match['original_id']
#         e_name = match['display_name']

#         # 3. TOOL EXECUTION (Retrieval)
#         if e_type == "Member":
#             if current_role == "Member" and e_id != current_user_id:
#                 return {"answer": "For security reasons, you can only access your own records."}
            
#             cur.execute("SELECT * FROM Member_Activity_Docs WHERE member_id = %s ORDER BY recorded_at DESC LIMIT 1", (e_id,))
#             log = cur.fetchone()
#             details = log['details'] if log else "No recent activity found."
#             return {"answer": f"Records for {e_name}: Last activity: {details}"}

#         if e_type == "Equipment":
#             cur.execute("SELECT status FROM Equipment_Assets WHERE asset_id = %s", (e_id,))
#             row = cur.fetchone()
#             status = row['status'] if row else "Unknown"
#             return {"answer": f"The {e_name} is currently {status}."}

#         # Managerial Analytics Route
#         if "revenue" in message.lower() or "profit" in message.lower():
#             if current_role != "Manager":
#                 return {"answer": "Only the Manager can access financial analytics."}
#             # Reuse your existing financial logic
#             cur.execute("SELECT SUM(final_price_paid) FROM Membership_Enrollments")
#             rev = cur.fetchone()['sum'] or 0
#             return {"answer": f"Our total membership revenue is ₹{rev}."}

#         return {"answer": f"I found {e_name} ({e_type}), but I'm not sure what you want to know about it."}

#     except Exception as e:
#         print(f"Chat Error: {e}")
#         return {"answer": "I'm having trouble accessing my knowledge base right now."}
#     finally:
#         cur.close()
#         conn.close()

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest, 
                #   current_user: User = Depends(get_current_user)
                  ):
    # Ignore the user_id in the body; use the ID from the secure JWT instead
    # config = {"configurable": {"thread_id": f"{current_user.id}_{req.session_id}"}}
    # input_data = {
    #     "messages": [HumanMessage(content=req.message)], 
    #     "user_context": {"id": current_user.id, "role": current_user.role}
    # }

    config = {"configurable": {"thread_id": f"{req.user_id}_{req.session_id}"}}
    input_data = {
        "messages": [HumanMessage(content=req.message)], 
        "user_context": {"id": req.user_id, "role": req.role}
    }
    
    try:
        # Calling the pre-compiled, cached agent
        result = gym_app.invoke(input_data, config=config)
        return {"answer": result["messages"][-1].content}
    except Exception as e:
        print(f"CHAT ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)








