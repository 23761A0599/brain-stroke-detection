from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.predict import router as predict_router

app = FastAPI(title="Brain Hemorrhage Detection API")

# Enable CORS for localhost:5173 and all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure outputs directory exists
outputs_dir = Path("outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)

# Mount outputs directory
app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)

@app.get("/")
def root():
    return {"message": "API Running"}

app.include_router(predict_router)