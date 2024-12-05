from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist
from main import main as advisor


class RequestBody(BaseModel):
    codes: conlist(str, max_length=300)  # list of str, max length 300
    risk_level: int = Field(..., ge=1, le=5)  # 1 <= risk_level <= 5
    investor_goal: int = Field(..., ge=1, le=4)  # 1 <= investor_goal <= 4


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용. 필요에 따라 특정 도메인으로 제한.
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용.
    allow_headers=["*"],  # 모든 HTTP 헤더 허용.
)


@app.get("/healthcheck")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/advisor")
def execute_roboadvisor(body: RequestBody):
    return advisor(body.codes, body.risk_level, body.investor_goal)
