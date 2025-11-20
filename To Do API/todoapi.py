# POST: Create tasks with details like title, description(optional), status, priority, category, due date, created time, id
# GET: Search task by id, status, priority, category, due date, title
# GET: Get task statistics, specific task
# PUT: Full update for a task
# PATCH: update timestamps, status (Partial updation)
# DELETE: Delete the task - permanently or archive

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel
import shutil
import os

app = FastAPI(title="To Do API", description="An API for to-do", version="1.0.0")
tasks = []  # Create in-memory storage to store tasks

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskCategory(Enum):
    WORK = "work"
    PERSONAL = "personal"
    URGENT = "urgent"

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    category: TaskCategory = TaskCategory.PERSONAL
    due_date: Optional[datetime] = None

class Task(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    category: TaskCategory = TaskCategory.PERSONAL
    due_date: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    attachments: List[str] = [] # Stores filenames

# Databsase dependency
def get_database():
    return tasks

def get_current_user(api_key: str = Header(None)):
    if api_key != "secret-key-123":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"user_id": 1, "username": "demo_user"}

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request, call_next):
    start_time=time.time()
    response=await call_next(request)
    process_time = time.time() - start_time
    print(f"Request: {request.method} {request.url} took {process_time:.2f}s")
    return response

# Root Endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to Todo API",
        "docs": "/docs",
        "version": "1.0.0"
    }

# Create tasks with details like id, title, description(optional), status, priority, category, due date, created time
@app.post("/tasks/")
def create_task(task_data: TaskCreate, db: list = Depends(get_database), current_user: dict = Depends(get_current_user)): 
    new_id = len(tasks) + 1
    new_task = Task(
        id=new_id,
        created_at=datetime.now(),
        title=task_data.title,
        description=task_data.description,
        status=task_data.status,
        priority=task_data.priority,
        category=task_data.category,
        due_date=task_data.due_date,
        attachments=[]
    )
    tasks.append(new_task)
    return new_task

# Get all the tasks
@app.get("/tasks/")
def get_all_tasks(): 
    return {"tasks": tasks, "count": len(tasks)}

# Search task through id
@app.get("/tasks/{id}")
def get_specific_task_through_id(id: int):
    for task in tasks:
        if task.id == id:
            return task
        
    raise HTTPException(status_code=404, detail="Task not found")

# Full task update
@app.put("/tasks/{id}")
def full_task_update(id: int, updated_data: TaskCreate):
    for index, task in enumerate(tasks):
        if task.id == id:
            tasks[index] = Task(id=task.id, created_at=task.created_at, updated_at=datetime.now(), title=updated_data.title, description=updated_data.description, status=updated_data.status, priority=updated_data.priority, category=updated_data.category, due_date=updated_data.due_date)
            return tasks[index]
    raise HTTPException(status_code=404, detail="Task not found")

# Partial Update
@app.patch("/tasks/{id}")
def partial_task_update(id: int, updated_data: TaskCreate):
    for index, task in enumerate(tasks):
        if task.id == id:
            tasks[index] = Task(id = task.id, created_at = task.created_at, updated_at = datetime.now(), title = updated_data.title if updated_data.title else task.title, description=updated_data.description if updated_data.description is not None else task.description, status=updated_data.status if updated_data.status else task.status, priority=updated_data.priority if updated_data.priority else task.priority, category=updated_data.category if updated_data.category else task.category, due_date=updated_data.due_date if updated_data.due_date is not None else task.due_date)
            return tasks[index]
    raise HTTPException(status_code=404, detail="Task not found")

# DELETE: Delete the task - permanently or archive
# Archive (Soft delete)
@app.delete("/tasks/{id}")
def archive_task(id: int):
    for index, task in enumerate(tasks):
        if task.id == id:
            tasks[index] = Task(id=task.id, created_at=task.created_at, updated_at=datetime.now(), title = task.title, description=task.description, status=TaskStatus.ARCHIVED, priority=task.priority, category=task.category, due_date=task.due_date)
            return {"message": f"Task {id} archived successfully."}
    raise HTTPException(status_code=404, detail="Task not found")

# Delete Permanently
@app.delete("/tasks/{id}/permanent")
def permanent_delete(id: int):
    for index, task in enumerate(tasks):
        if task.id == id:
            del tasks[index]
            return {"message": f"Task {id} deleted successfully."}
    raise HTTPException(status_code=404, detail="Task not found")

# Get all tasks using filters
@app.get("/tasks")
def get_filtered_tasks(status: Optional[TaskStatus] = None, priority: Optional[TaskPriority] = None, category: Optional[TaskCategory] = None, search: Optional[str] = None):
    filtered_tasks = tasks.copy()
    if status:
        filtered_tasks = [task for task in filtered_tasks if task.status == status]
    if priority:
        filtered_tasks = [task for task in filtered_tasks if task.priority == priority]
    if category:
        filtered_tasks = [task for task in filtered_tasks if task.category == category]
    if search:
        filtered_tasks = [task for task in filtered_tasks if search.lower() in task.title.lower() or (task.description and search.lower() in task.description.lower())]
    return {"Tasks": filtered_tasks, "count": len(filtered_tasks)}

# Getting statistics
@app.get("/stats")
def task_stats():
    # Status - pending
    pending_tasks = len([task for task in tasks if task.status == TaskStatus.PENDING])
    # Status - in progress
    inprogress_tasks = len([task for task in tasks if task.status == TaskStatus.IN_PROGRESS])
    # Status - completed
    completed_tasks = len([task for task in tasks if task.status == TaskStatus.COMPLETED])
    # Status - archived
    archived_tasks = len([task for task in tasks if task.status == TaskStatus.ARCHIVED])
    # Priority - low
    low_priority_tasks = len([task for task in tasks if task.priority == TaskPriority.LOW])
    # Priority - medium
    medium_priority_tasks = len([task for task in tasks if task.priority == TaskPriority.MEDIUM])
    # Priority - high
    high_priority_tasks = len([task for task in tasks if task.priority == TaskPriority.HIGH])
    # Priority - critical
    critical_priority_tasks = len([task for task in tasks if task.priority == TaskPriority.CRITICAL])
    # Category - work
    work_tasks = len([task for task in tasks if task.category == TaskCategory.WORK])
    # Category - personal
    personal_tasks = len([task for task in tasks if task.category == TaskCategory.PERSONAL])
    # Category - urgent
    urgent_tasks = len([task for task in tasks if task.category == TaskCategory.URGENT])
    # Completion Rate
    active_tasks = pending_tasks + inprogress_tasks + completed_tasks # excluding archived tasks
    if active_tasks > 0:
        completion_rate = round((completed_tasks * 100 / active_tasks), 2)
    else :
        completion_rate = 0
    return {
        "Pending Tasks": pending_tasks,
        "In Progress Tasks": inprogress_tasks,
        "Completed Tasks": completed_tasks,
        "Archived Tasks": archived_tasks,
        "Low Priority Tasks": low_priority_tasks,
        "Medium Priority Tasks": medium_priority_tasks, 
        "High Priority Tasks": high_priority_tasks, 
        "Critical Priority Tasks": critical_priority_tasks,
        "Work Category Tasks": work_tasks,
        "Personal Category Tasks": personal_tasks, 
        "Urgent Category Tasks": urgent_tasks,
        "Completion Rate": completion_rate
    }

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/tasks/{task_id}/upload")
async def upload_task_files(task_id: int, files: List[UploadFile] = File()):
    for index, task in enumerate(tasks):
        if task.id == task_id:
            saved_files = []
            for file in files:
                filename = f"{task_id}_{file.filename}"
                file_path = os.path.join(UPLOAD_DIR, filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                saved_files.append(filename)

            updated_attachments = task.attachments + saved_files
            tasks[index] = Task(
                id=task.id,
                created_at=task.created_at,
                updated_at=datetime.now(),
                title=task.title,
                description=task.description,
                status=task.status,
                priority=task.priority,
                category=task.category,
                due_date=task.due_date,
                attachments=updated_attachments
            )

            return {
                "message": f"Uploaded {len(saved_files)} files to task {task_id}",
                "saved_files": saved_files,
                "total_attachments": len(updated_attachments) 
            }
    raise HTTPException(status_code=404, detail="Task not found")

@app.post("/upload-with-data/")
async def upload_file_with_data(file: UploadFile = File(), category: str = Form(), description: str =Form()):
    return{
        "file_info": {
            "name": file.filename, "type": file.content_type
        }, 
        "metadata": {
            "category": category, 
            "description": description
        }
    }

@app.get("/tasks/{task_id}/files/{filename}")
def download_task_file(task_id: int, filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not filename.startswith(f"{task_id}_"):
        raise HTTPException(status_code=403, detail="Access denied")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")
