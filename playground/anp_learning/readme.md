myAgent 作为 ANP 服务端把内部 rag-assistant 包装成 ANP Agent。
生成 ad.json，描述 Agent 身份和能力。
生成 interface.json，声明 ask() 方法和参数。
通过 /rpc 接收外部调用。
外部客户端成功发现并调用了 RAG。

myAgent 作为 ANP 客户端为 myAgent 生成 DID 客户端身份和私钥。
独立启动 8090 LearningAgent，模拟第三方服务。
myAgent 获取其 ad.json。
根据 interface.json 自动发现 hello(name)。
通过 JSON-RPC 调用 hello()。
成功获得第三方 Agent 返回值。

