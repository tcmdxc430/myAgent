"""LeeNexus custom A2A entry point for the Git status agent."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from schema import UserInput

router = APIRouter(prefix="/a2a", tags=["A2A"])


class A2AParameters(BaseModel):
    query: str = Field(min_length=1)


class A2ARequest(BaseModel):
    parameters: A2AParameters


@router.post("/myagent")
async def invoke_myagent(request: A2ARequest) -> dict[str, str]:
    from service.service import invoke

    answer = await invoke(UserInput(message=request.parameters.query), "leenexus-git-agent")
    return {"response": str(answer.content)}
