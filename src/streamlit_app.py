import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

# 一个用于通过简单的聊天界面与 langgraph 智能体交互的 Streamlit 应用。
# 该应用有三个主要功能，均为异步运行：

# - main() - 设置 streamlit 应用及其高层结构
# - draw_messages() - 绘制一组聊天消息 - 无论是回放现有消息还是流式传输新消息。
# - handle_feedback() - 绘制反馈组件并记录用户反馈。

# 该应用大量使用 AgentClient 与智能体的 FastAPI 端点进行交互。


APP_TITLE = "AI 智能体"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("正在连接智能体服务..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"连接智能体服务失败 ({agent_url}): {e}")
            st.markdown("服务可能正在启动中，请几秒后重试。")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    # Initialize voice manager (once per session)
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("未找到该会话 ID 的历史记录。")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # 配置选项
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "对话创造更多可能"
        ""

        if st.button(":material/chat: 新对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # Clear saved audio when starting new chat
            if "last_audio" in st.session_state:
                del st.session_state.last_audio
            st.rerun()

        with st.expander(":material/settings: 设置", expanded=False):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("使用的模型", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "使用的智能体",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("流式输出结果", value=True)
            # Audio toggle with callback: clears cached audio when toggled off
            enable_audio = st.toggle(
                "启用语音生成",
                value=True,
                disabled=not voice or not voice.tts,
                help="在 .env 中配置 VOICE_TTS_PROVIDER 以启用"
                if not voice or not voice.tts
                else None,
                on_change=lambda: st.session_state.pop("last_audio", None)
                if not st.session_state.get("enable_audio", True)
                else None,
                key="enable_audio",
            )

            # Display user ID (for debugging or user information)
            st.text_input("用户 ID (只读)", value=user_id, disabled=True)

        @st.dialog("分享/恢复对话")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # Include both thread_id and user_id in the URL for sharing to maintain user identity
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.markdown(f"**对话 URL:**\n```text\n{chat_url}\n```")
            st.info("复制上方 URL 以分享或重新访问此对话")

        if st.button(":material/upload: 分享/恢复对话", use_container_width=True):
            share_chat_dialog()

        with st.popover(":material/bookmark_add: 图文链接导入", use_container_width=True):
            st.caption("导入小红书、网页文章等图文内容到 RAG 知识库")
            article_url = st.text_input("图文文章链接", key="article_ingest_url")
            force_refresh = st.checkbox("强制刷新已导入内容", value=False)
            col_import, col_login = st.columns(2)

            if col_import.button("导入", use_container_width=True):
                if not article_url.strip():
                    st.warning("请先填写图文文章链接。")
                else:
                    with st.spinner("正在抓取并写入知识库..."):
                        try:
                            st.session_state.article_ingest_result = (
                                await agent_client.aingest_article(
                                    article_url.strip(), force_refresh=force_refresh
                                )
                            )
                        except AgentClientError as e:
                            st.error(f"导入失败: {e}")

            if col_login.button("登录小红书", use_container_width=True):
                try:
                    login_response = await agent_client.aopen_xhs_login()
                    st.info(login_response.message)
                except AgentClientError as e:
                    st.error(f"打开登录窗口失败: {e}")

            result = st.session_state.get("article_ingest_result")
            if result:
                match result.status:
                    case "success":
                        st.success(f"导入成功: {result.title or result.source_url}")
                    case "partial_success":
                        st.warning(f"导入完成，但 {result.ocr_failed_count} 张图片 OCR 失败。")
                    case "login_required":
                        st.warning("需要先登录对应平台。小红书可点击上方“登录小红书”后重试。")
                    case "skipped":
                        st.info(result.message)
                    case _:
                        st.error(result.message)
                if result.chunk_count:
                    st.caption(f"已写入 {result.chunk_count} 个 RAG chunk。")

        # 如果当前选择的是数据分析智能体，显示文件上传控件
        if agent_client.agent == "data-analyst-assistant":
            st.markdown("---")
            st.subheader("📊 上传分析数据")
            uploaded_file = st.file_uploader("上传 CSV 格式数据文件", type=["csv"])
            if uploaded_file is not None:
                # 确保 data 目录存在
                os.makedirs("data", exist_ok=True)
                # 保存为指定的文件名 data/uploaded_data.csv
                with open("data/uploaded_data.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ 文件 '{uploaded_file.name}' 上传成功！你可以开始在聊天框提问进行分析了。")



    # 绘制现有消息
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "你好！我是一个简单的聊天机器人。有什么我可以帮你的吗？"
            case "interrupt-agent":
                WELCOME = "你好！我是中断智能体。告诉我你的生日，我会预测你的性格！"
            case "research-assistant":
                WELCOME = "你好！我是 AI 驱动的研究助手，拥有网页搜索和计算器功能。问我任何问题吧！"
            case "rag-assistant":
                WELCOME = """你好！我是 AI 驱动的公司政策和人力资源助手，可以访问 AcmeTech 的员工手册。
                我可以帮你查找有关福利、远程办公、休假政策、公司价值观等信息。问我任何问题吧！"""
            case "data-analyst-assistant":
                WELCOME = """你好！我是你的专属 AI 数据分析科学家。⚙️
                请先在左侧边栏上传你的 CSV 文件，然后告诉我你想要进行什么分析、画什么图表，或者让我直接为你做一份全面分析！"""
            case "technical-report-agent":
                WELCOME = """你好！我是技术报告写作智能体。
                我会基于已导入的网页 URL 或文档知识库，按规范结构生成政策合规、表达稳妥的技术报告。"""
            case _:
                WELCOME = "你好！我是 AI 智能体。问我任何问题吧！"

        with st.chat_message("ai", avatar="media/avator.png"):
            st.write(WELCOME)

    # draw_messages() 期望一个消息的异步迭代器
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # 为最后一条 AI 消息渲染保存的音频（如果存在）
    # 这确保了音频在 st.rerun() 调用后依然存在
    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # 如果用户提供了新输入，生成新消息
    # 如果可用，使用语音管理器，否则回退到常规输入
    # 必填：在应用 .env（不是服务 .env）中设置 VOICE_STT_PROVIDER, VOICE_TTS_PROVIDER, OPENAI_API_KEY 以启用语音功能。
    if voice:
        user_input = voice.get_chat_input()
    else:
        user_input = st.chat_input()

    if user_input:
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human", avatar="👤").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
                # 为流式响应生成 TTS 音频
                # 注意：draw_messages() 将最终消息存储在 st.session_state.messages 中
                # 并将容器引用存储在 st.session_state.last_message 中
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # 仅为有内容的 AI 响应生成音频
                    if last_msg.type == "ai" and last_msg.content:
                        # 使用 audio_only=True，因为文本已经由 draw_messages() 流式传输
                        voice.render_message(
                            last_msg.content,
                            container=st.session_state.last_message,
                            audio_only=True,
                        )
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                # 渲染带有可选语音的 AI 响应
                with st.chat_message("ai", avatar="media/avator.png"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
            st.rerun()  # 清除陈旧的容器
        except AgentClientError as e:
            st.error(f"生成响应时出错: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    绘制一组聊天消息 - 无论是回放现有消息还是流式传输新消息。

    该函数具有处理流式令牌和工具调用的额外逻辑。
    - 使用占位符容器在流式令牌到达时进行渲染。
    - 使用状态容器渲染工具调用。跟踪工具输入和输出并相应地更新状态容器。

    该函数还需要在会话状态中跟踪最后一条消息容器，因为后续消息可以绘制到同一个容器中。
    这也用于在最新的聊天消息中绘制反馈组件。

    参数:
        messages_agen: 要绘制的消息的异步迭代器。
        is_new: 消息是否为新消息。
    """

    # 跟踪最后一条消息容器
    last_message_type = None
    st.session_state.last_message = None

    # 中间流式令牌的占位符
    streaming_content = ""
    streaming_placeholder = None

    # 遍历消息并绘制它们
    while msg := await anext(messages_agen, None):
        # str 消息代表正在流式传输的中间令牌
        if isinstance(msg, str):
            # 如果占位符为空，这是正在流式传输的新消息的第一个令牌。我们需要进行设置。
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai", avatar="media/avator.png")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"非预期的消息类型: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # 来自用户的消息，最简单的情况
            case "human":
                last_message_type = "human"
                st.chat_message("human", avatar="👤").write(msg.content)

            # 来自智能体的消息是最复杂的情况，因为我们需要处理流式令牌和工具调用。
            case "ai":
                # 如果我们正在渲染新消息，将消息存储在会话状态中
                if is_new:
                    st.session_state.messages.append(msg)

                # 如果最后一条消息类型不是 AI，创建一个新的聊天消息
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai", avatar="media/avator.png")

                with st.session_state.last_message:
                    # 如果消息有内容，将其写出。
                    # 重置流式变量以准备下一条消息。
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # 为每个工具调用创建一个状态容器，并按 ID 存储状态容器，以确保结果映射到正确的状态容器。
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            # 为传输与常规工具调用使用不同的标签
                            if "transfer_to" in tool_call["name"]:
                                label = f"""💼 子智能体: {tool_call["name"]}"""
                            else:
                                label = f"""🛠️ 工具调用: {tool_call["name"]}"""

                            status = st.status(
                                label,
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status

                        # 每个工具调用期望一个 ToolMessage。
                        transfer_tool_call = None
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                transfer_tool_call = tool_call
                                break

                        if transfer_tool_call:
                            status = call_results[transfer_tool_call["id"]]
                            status.update(expanded=True)
                            returned_msg = await handle_sub_agent_msgs(messages_agen, status, is_new)
                            # 重置所有跟踪变量以强制使用新容器
                            last_message_type = None
                            st.session_state.last_message = None
                            streaming_content = ""
                            streaming_placeholder = None

                            # 如果返回了人类消息，现在绘制它
                            if returned_msg:
                                last_message_type = "human"
                                st.chat_message("human", avatar="👤").write(returned_msg.content)
                                if is_new:
                                    st.session_state.messages.append(returned_msg)
                            continue

                        for tool_call in msg.tool_calls:
                            # 只有非传输工具调用会到达这里
                            status = call_results[tool_call["id"]]
                            status.write("输入:")
                            status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"非预期的聊天消息类型: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # 如果是新消息则记录，并使用结果更新正确的状态容器
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("输出:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # bg-task-agent 使用的 CustomData 示例
                # 参见:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("从智能体接收到非预期的 CustomData 消息")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # 如果出现非预期的消息类型，记录错误并停止
            case _:
                st.error(f"非预期的聊天消息类型: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"记录反馈时出错: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("反馈已记录", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.

    Enhanced to support nested multi-agent hierarchies with handoff back messages.

    Args:
        messages_agen: Async generator of messages
        status: the status container for the current agent
        is_new: Whether messages are new or replayed
    """
    nested_popovers = {}

    # looking for the transfer Success tool call message
    while True:
        first_msg = await anext(messages_agen)
        if not isinstance(first_msg, str):
            break
    if is_new:
        st.session_state.messages.append(first_msg)

    # Continue reading until we get an explicit handoff back or the stream ends
    while True:
        # Read next message
        try:
            sub_msg = await anext(messages_agen)
        except StopAsyncIteration:
            break

        # Skip tokens (strings) and only process ChatMessage objects
        if isinstance(sub_msg, str):
            continue

        # If we encounter a human message, it means the sub-agent's turn is over
        # and a new interaction has started. Return the message so the main loop can draw it.
        if sub_msg.type == "human":
            return sub_msg

        if is_new:
            st.session_state.messages.append(sub_msg)

        # Handle tool results with nested popovers
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**输出:**")
            popover.write(sub_msg.content)
            continue

        # Handle transfer_back_to tool calls - these indicate a sub-agent is returning control
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # Process transfer_back_to tool calls
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # Read the corresponding tool result
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # After processing transfer back, we're done with this agent
            if status:
                status.update(state="complete")
            break

        # Display content and tool calls in the same nested status
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # Check if this is a nested transfer/delegate
                    if "transfer_to" in tc["name"]:
                        # Create a nested status container for the sub-agent
                        nested_status = status.status(
                            f"""💼 子智能体: {tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # Recursively handle sub-agents of this sub-agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # Regular tool call - create popover
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**工具:** {tc['name']}")
                        popover.write("**输入:**")
                        popover.write(tc["args"])
                        # Store the popover reference using the tool call ID
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
