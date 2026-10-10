"""Agent that calls local Git, calculator, and work order tools through LeeNexus."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph

from core import settings

GIT_STATUS_TOOL = "git-mcp-local-git-status"
CALCULATOR_TOOL = "a2a-leenexus-calculator-demo"
WORK_ORDER_TOOL = "leenexus-submit-work-order"
BEIJING_TIMEZONE = timezone(timedelta(hours=8))


class LeeNexusPermissionDenied(RuntimeError):
    """LeeNexus rejected a tool call during authorization."""


async def _call_leenexus_tool(name: str, arguments: dict[str, str]) -> str:
    """Invoke a registered LeeNexus tool and return its text result."""
    if not settings.LEENEXUS_TOKEN:
        raise RuntimeError("LEENEXUS_TOKEN is not configured")

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    timeout = 330 if name == WORK_ORDER_TOOL else 30
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.LEENEXUS_URL.rstrip('/')}/rpc",
            json=request,
            headers={"Authorization": f"Bearer {settings.LEENEXUS_TOKEN.get_secret_value()}"},
        )
        response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        if payload["error"].get("code") == -32003:
            message = payload["error"].get("message", "")
            if "灵钥" in message or "智能体" in message:
                raise LeeNexusPermissionDenied(message)
            raise LeeNexusPermissionDenied("当前令牌或团队没有执行该工具的权限")
        raise RuntimeError(f"LeeNexus tool call failed: {payload['error']}")
    result = payload.get("result", {})
    if result.get("isError"):
        raise RuntimeError(f"LeeNexus tool {name} failed: {result}")
    return "\n".join(part.get("text", "") for part in result.get("content", [])) or str(result)


@tool
async def leenexus_git_status() -> str:
    """Read the current myAgent repository's Git status using LeeNexus."""
    repo_path = str(Path(__file__).resolve().parents[2])
    return await _call_leenexus_tool(GIT_STATUS_TOOL, {"repo_path": repo_path})


@tool
async def leenexus_calculator(query: str) -> str:
    """Send a calc: expression to the registered LeeNexus calculator."""
    return await _call_leenexus_tool(CALCULATOR_TOOL, {"query": query})


@tool
async def leenexus_submit_work_order(title: str, details: str) -> str:
    """Submit one work order after Lingya reviews the exact request."""
    return await _call_leenexus_tool(WORK_ORDER_TOOL, {"title": title, "details": details, "request_id": uuid4().hex})


def _format_work_order_result(raw_result: str) -> str:
    """Present a confirmed work order without claiming that the inspection ran."""
    try:
        result = json.loads(raw_result)
    except json.JSONDecodeError:
        return "工单提交结果无法确认。请先核对工单记录，避免重复提交。"

    if not isinstance(result, dict):
        return "工单提交结果无法确认。请先核对工单记录，避免重复提交。"

    work_order_id = result.get("id")
    title = result.get("title")
    status = result.get("status")
    if not isinstance(work_order_id, str) or not work_order_id or not isinstance(title, str) or not title or status not in {"created", "existing"}:
        return "工单提交结果无法确认。请先核对工单记录，避免重复提交。"

    heading = "工单已创建" if status == "created" else "工单已存在，本次没有重复创建"
    lines = [heading, f"内容：{title}", f"工单编号：{work_order_id}"]
    created_at = result.get("created_at")
    if isinstance(created_at, str):
        try:
            timestamp = datetime.fromisoformat(created_at)
            if timestamp.tzinfo is not None:
                lines.append(f"创建时间：{timestamp.astimezone(BEIJING_TIMEZONE):%Y-%m-%d %H:%M:%S}（北京时间）")
        except ValueError:
            pass
    lines.append("这一步仅登记检查请求，尚未实际检查设备。")
    return "\n".join(lines)


async def route_leenexus_request(state: MessagesState) -> dict:
    """Route calculator and work order requests to LeeNexus."""
    message = state["messages"][-1]
    query = message.content if isinstance(message.content, str) else ""
    try:
        if query.strip().lower().startswith("calc:"):
            answer = await leenexus_calculator.ainvoke({"query": query.strip()})
        elif query.strip().lower().startswith("workorder:"):
            details = query.strip().split(":", 1)[1].strip()
            if not details:
                answer = "请在 workorder: 后写明工单内容。"
            else:
                result = await leenexus_submit_work_order.ainvoke({"title": details[:160], "details": details})
                answer = _format_work_order_result(result)
        else:
            answer = await leenexus_git_status.ainvoke({})
    except LeeNexusPermissionDenied as exc:
        target = "工单" if query.strip().lower().startswith("workorder:") else "计算器"
        answer = f"调用被拒绝：{exc}。{target}没有执行。"
    return {"messages": [AIMessage(content=answer)]}


graph = StateGraph(MessagesState)
graph.add_node("route_leenexus_request", route_leenexus_request)
graph.set_entry_point("route_leenexus_request")
graph.add_edge("route_leenexus_request", END)
leenexus_git_agent = graph.compile()
