import os
import json
import re
import shutil
import asyncio
import urllib.parse
import hashlib
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timezone, timedelta
from PIL import Image
from google import genai
from dotenv import load_dotenv
import uuid

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("No API key found in .env file.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

app = FastAPI(title="PocketSmart: AI Budget Planner")

os.makedirs("templates", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_12345")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

class Token(BaseModel):
    access_token: str
    token_type: str

class UserInDB(BaseModel):
    username: str
    email: Optional[str] = None
    hashed_password: Optional[str] = None

class UserSession(BaseModel):
    username: str
    login_time: datetime
    last_activity: datetime
    token: str
    user_data: Dict[str, Any] = {}

class RecommendationHistory(BaseModel):
    id: str
    timestamp: str
    recommendation_type: str
    input_summary: Any
    result_summary: Any
    full_result: Any = None

users_db: Dict[str, dict] = {}
active_sessions: Dict[str, UserSession] = {}
user_recommendations: Dict[str, List[RecommendationHistory]] = {}
blacklisted_tokens = set()

def get_password_hash(password: str) -> str:
    salt = "pocketsmart_salt_"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return get_password_hash(plain_password) == hashed_password

def authenticate_user(db: dict, username: str, password: str):
    user = db.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return UserInDB(username=user["username"], email=user.get("email"))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    return request.cookies.get("access_token")

async def get_current_user(request: Request, token: Optional[str] = Depends(get_token)) -> Optional[UserInDB]:
    if not token or token in blacklisted_tokens:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return UserInDB(username=username, email=users_db.get(username, {}).get("email"))
    except JWTError:
        return None

async def get_current_active_user(user: Optional[UserInDB] = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user

def save_upload_file(upload_file: UploadFile) -> str:
    file_path = os.path.join("static/uploads", f"{uuid.uuid4()}_{upload_file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

def save_to_history(username: str, recommendation_type: str, input_data: Any, result: Any):
    if username not in user_recommendations:
        user_recommendations[username] = []
    history_entry = RecommendationHistory(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        recommendation_type=recommendation_type,
        input_summary=input_data,
        result_summary=result,
        full_result=result
    )
    user_recommendations[username].append(history_entry)

@app.on_event("startup")
async def setup_session_cleanup():
    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.now(timezone.utc)
            expired_sessions = [
                username for username, session in active_sessions.items()
                if (current_time - session.last_activity).total_seconds() > 1800
            ]
            for username in expired_sessions:
                if username in active_sessions:
                    del active_sessions[username]
            await asyncio.sleep(300)

    asyncio.create_task(cleanup_expired_sessions())

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

    existing_user_data = {}
    if user.username in active_sessions:
        existing_user_data = active_sessions[user.username].user_data
        old_token = active_sessions[user.username].token
        blacklisted_tokens.add(old_token)

    now = datetime.now(timezone.utc)
    active_sessions[user.username] = UserSession(
        username=user.username,
        login_time=now,
        last_activity=now,
        token=access_token,
        user_data=existing_user_data
    )

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax"
    )
    return response

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None, success: Optional[str] = None):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": error, "success": success}
    )

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = authenticate_user(users_db, username, password)
    if not user:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Incorrect username or password. Please try again."}
        )

    access_token = create_access_token(data={"sub": username})
    now = datetime.now(timezone.utc)
    active_sessions[username] = UserSession(
        username=username, login_time=now, last_activity=now, token=access_token, user_data={}
    )
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"error": error}
    )

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: Optional[str] = Form(None)
):
    if confirm_password and password != confirm_password:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Passwords do not match. Please re-enter."}
        )

    if username in users_db:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Username is already registered. Please choose another."}
        )

    for u in users_db.values():
        if u.get("email") == email:
            return templates.TemplateResponse(
                request=request,
                name="register.html",
                context={"error": "This email address is already registered!"}
            )

    users_db[username] = {
        "username": username,
        "email": email,
        "hashed_password": get_password_hash(password)
    }

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"success": "Registration successful! You can now sign in with your credentials."}
    )

@app.post("/logout")
async def logout(request: Request):
    token = await get_token(request)
    if token:
        blacklisted_tokens.add(token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username and username in active_sessions:
                del active_sessions[username]
        except JWTError:
            pass
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"user": current_user}
    )

class HomeBudgetInput(BaseModel):
    room_type: str = "Living Room"
    total_budget: float
    preferences: Optional[str] = "Modern"
    num_lights: Optional[int] = 4
    num_fans: Optional[int] = 1
    num_furniture: Optional[int] = 2
    num_dining_tables: Optional[int] = 0

class PartyBudgetInput(BaseModel):
    party_type: str = "Birthday"
    num_guests: int = 10
    total_budget: float
    venue_type: Optional[str] = None
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = False
    additional_requirements: Optional[str] = None

class JewelryBudgetInput(BaseModel):
    occasion: str = "Wedding"
    total_budget: float
    preferences: Optional[str] = "Traditional Gold/Diamond"

def extract_json_from_response(text: str) -> dict:
    try:
        clean_text = re.sub(r"^```json\s*", "", text.strip())
        clean_text = re.sub(r"\s*```$", "", clean_text)
        return json.loads(clean_text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}

def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    try:
        prompt = f"""
Generate home interior recommendations for India with a total budget of INR {budget_input.total_budget:.2f}.
Room Type: {budget_input.room_type}
Preferences: {budget_input.preferences}

Format strictly as JSON:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "home_recommendations": [
    {{"category": "Furniture", "item_name": "Sofa", "estimated_price": 0.0, "search_terms": "wooden sofa"}}
  ],
  "remaining_budget": 0.0,
  "tips": []
}}
"""
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        result = extract_json_from_response(response.text)
        for item in result.get("home_recommendations", []):
            st = item.get("search_terms", "")
            if st:
                enc = urllib.parse.quote_plus(st)
                item["shopping_links"] = {
                    "amazon": f"https://www.amazon.in/s?k={enc}",
                    "flipkart": f"https://www.flipkart.com/search?q={enc}",
                    "ikea": f"https://www.ikea.com/in/en/search/?q={enc}"
                }
        return result
    except Exception as e:
        raise HTTPException(500, f"Error generating recommendations: {str(e)}")

def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    try:
        prompt = f"Party recommendations INR {budget_input.total_budget:.2f}, Type: {budget_input.party_type}"
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return extract_json_from_response(response.text)
    except Exception as e:
        raise HTTPException(500, f"Error generating recommendations: {str(e)}")

def get_jewelry_recommendations(budget_input: JewelryBudgetInput, image_path: Optional[str] = None) -> dict:
    try:
        base_prompt = f"Jewelry recommendations INR {budget_input.total_budget:.2f}, Occasion: {budget_input.occasion}"
        contents = [base_prompt]
        if image_path and os.path.exists(image_path):
            contents.append(Image.open(image_path))
        response = client.models.generate_content(model=MODEL_NAME, contents=contents)
        result = extract_json_from_response(response.text)
        for item in result.get("jewelry_recommendations", []):
            st = item.get("search_terms", "")
            if st:
                enc = urllib.parse.quote_plus(st)
                item["shopping_links"] = {
                    "amazon": f"https://www.amazon.in/s?k={enc}",
                    "flipkart": f"https://www.flipkart.com/search?q={enc}",
                    "tanishq": f"https://www.tanishq.co.in/search?q={enc}"
                }
        return result
    except Exception as e:
        raise HTTPException(500, f"Error generating recommendations: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, current_user: Optional[UserInDB] = Depends(get_current_user)):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"user": current_user}
    )

@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="home_planner.html",
        context={"user": current_user}
    )

@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="party_planner.html",
        context={"user": current_user}
    )

@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="jewelry_planner.html",
        context={"user": current_user}
    )

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"user": current_user}
    )

@app.post("/generate-home")
@app.post("/home-budget")
async def plan_home_budget(
    budget_input: HomeBudgetInput,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    result = get_home_recommendations(budget_input)
    save_to_history(
        username=current_user.username,
        recommendation_type="home",
        input_data=budget_input.model_dump(),
        result=result
    )
    return result

@app.post("/generate-party")
async def generate_party(
    budget_input: PartyBudgetInput,
    current_user: UserInDB = Depends(get_current_active_user)
):
    result = get_party_recommendations(budget_input)
    save_to_history(
        username=current_user.username,
        recommendation_type="party",
        input_data=budget_input.model_dump(),
        result=result
    )
    return result

@app.post("/generate-jewelry")
@app.post("/jewelry-budget")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    request: Request = None,
    current_user: UserInDB = Depends(get_current_active_user)
):
    budget_input = JewelryBudgetInput(
        total_budget=total_budget,
        occasion=occasion,
        preferences=preferences
    )

    image_path = None
    if image:
        image_path = save_upload_file(image)

    result = get_jewelry_recommendations(budget_input, image_path)

    input_data = budget_input.model_dump()
    if image:
        input_data["image"] = image.filename

    save_to_history(
        username=current_user.username,
        recommendation_type="jewelry",
        input_data=input_data,
        result=result
    )
    return result

@app.get("/recommendation-history")
async def get_recommendation_history(
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    if current_user.username not in user_recommendations:
        return {"history": []}

    history = sorted(
        user_recommendations[current_user.username],
        key=lambda x: x.timestamp,
        reverse=True
    )

    return {"history": [{
        "id": item.id,
        "timestamp": item.timestamp,
        "type": item.recommendation_type,
        "input": item.input_summary,
        "summary": item.result_summary
    } for item in history]}

@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(
    recommendation_id: str,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    if current_user.username not in user_recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found")

    for item in user_recommendations[current_user.username]:
        if item.id == recommendation_id:
            return {
                "id": item.id,
                "timestamp": item.timestamp,
                "type": item.recommendation_type,
                "input": item.input_summary,
                "full_result": item.full_result
            }

    raise HTTPException(status_code=404, detail="Recommendation not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)