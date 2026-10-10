"""Tests for routing myAgent conversations to LeeNexus tools."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agents.leenexus_git_agent import LeeNexusPermissionDenied, route_leenexus_request


@pytest.mark.asyncio
async def test_calc_query_invokes_calculator() -> None:
    """A calc: message invokes the registered calculator with its full query."""
    with (
        patch("agents.leenexus_git_agent.leenexus_calculator") as calculator,
        patch("agents.leenexus_git_agent.leenexus_git_status") as git_status,
    ):
        calculator.ainvoke = AsyncMock(return_value="52")
        result = await route_leenexus_request(
            {"messages": [HumanMessage(content="  calc: 5*10+2  ")]}
        )

    calculator.ainvoke.assert_awaited_once_with({"query": "calc: 5*10+2"})
    git_status.ainvoke.assert_not_called()
    assert result["messages"][0].content == "52"


@pytest.mark.asyncio
async def test_other_queries_keep_git_status_route() -> None:
    """Existing Git status requests continue to invoke the Git tool."""
    with (
        patch("agents.leenexus_git_agent.leenexus_calculator") as calculator,
        patch("agents.leenexus_git_agent.leenexus_git_status") as git_status,
    ):
        git_status.ainvoke = AsyncMock(return_value="Repository status: clean")
        result = await route_leenexus_request(
            {"messages": [HumanMessage(content="查询 myAgent 仓库状态")]}
        )

    git_status.ainvoke.assert_awaited_once_with({})
    calculator.ainvoke.assert_not_called()
    assert result["messages"][0].content == "Repository status: clean"


@pytest.mark.asyncio
async def test_work_order_query_uses_approved_tool_route() -> None:
    """A workorder: message requests the registered write tool."""
    with patch("agents.leenexus_git_agent.leenexus_submit_work_order") as work_order:
        work_order.ainvoke = AsyncMock(return_value='{"id":"demo-123","title":"检查设备故障","created_at":"2026-10-09T03:33:38+00:00","status":"created"}')
        result = await route_leenexus_request({"messages": [HumanMessage(content="workorder: 检查设备故障")]})
    work_order.ainvoke.assert_awaited_once_with({"title": "检查设备故障", "details": "检查设备故障"})
    assert result["messages"][0].content == (
        "工单已创建\n"
        "内容：检查设备故障\n"
        "工单编号：demo-123\n"
        "创建时间：2026-10-09 11:33:38（北京时间）\n"
        "这一步仅登记检查请求，尚未实际检查设备。"
    )


@pytest.mark.asyncio
async def test_rejected_work_order_reports_no_execution() -> None:
    """A human denial reports that no work order was submitted."""
    with patch("agents.leenexus_git_agent.leenexus_submit_work_order") as work_order:
        work_order.ainvoke = AsyncMock(side_effect=LeeNexusPermissionDenied("灵钥拒绝调用（DENY）"))
        result = await route_leenexus_request({"messages": [HumanMessage(content="workorder: 检查设备故障")]})
    assert "灵钥拒绝" in result["messages"][0].content
    assert "工单没有执行" in result["messages"][0].content


@pytest.mark.asyncio
async def test_denied_calculator_call_returns_clear_message() -> None:
    """The conversation explains an authorization denial without a traceback."""
    with patch("agents.leenexus_git_agent.leenexus_calculator") as calculator:
        calculator.ainvoke = AsyncMock(side_effect=LeeNexusPermissionDenied("denied"))
        result = await route_leenexus_request(
            {"messages": [HumanMessage(content="calc: 5*10+2")]}
        )

    assert "调用被拒绝：denied" in result["messages"][0].content
    assert "计算器没有执行" in result["messages"][0].content
