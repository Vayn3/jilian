# -*- coding: utf-8 -*-
"""
LLM客户端模块 - 千问大模型流式调用
支持流式输出、多轮对话、RAG接口预留
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Dict, List, Optional, Any

from openai import AsyncOpenAI

from config import get_config, LLMConfig, RAGConfig

logger = logging.getLogger(__name__)


# ================== RAG接口定义（预留） ==================
class Document:
    """检索文档"""
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, score: float = 0.0):
        self.content = content
        self.metadata = metadata or {}
        self.score = score


class RAGInterface(ABC):
    """RAG检索增强接口（抽象基类）"""
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
        
        Returns:
            相关文档列表
        """
        pass
    
    @abstractmethod
    async def build_context(self, documents: List[Document]) -> str:
        """
        构建上下文
        
        Args:
            documents: 检索到的文档
        
        Returns:
            构建的上下文字符串
        """
        pass
    
    async def enhance_query(self, query: str, top_k: int = 3) -> str:
        """
        增强查询（检索+构建上下文）
        
        Args:
            query: 原始查询
            top_k: 检索文档数
        
        Returns:
            增强后的提示
        """
        documents = await self.retrieve(query, top_k)
        if not documents:
            return query
        
        context = await self.build_context(documents)
        enhanced_prompt = f"""基于以下参考资料回答用户问题:

参考资料:
{context}

用户问题: {query}

请根据参考资料回答，如果参考资料中没有相关信息，请基于通用知识回答。"""
        
        return enhanced_prompt


class DummyRAG(RAGInterface):
    """空RAG实现（占位用）"""
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        return []
    
    async def build_context(self, documents: List[Document]) -> str:
        return ""


class SimpleRAG(RAGInterface):
    """
    简单RAG实现示例（预留接口）
    实际使用时可替换为向量数据库实现
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or get_config().rag
        self.knowledge_base: List[Document] = []
    
    def add_document(self, content: str, metadata: Optional[Dict] = None) -> None:
        """添加文档到知识库"""
        self.knowledge_base.append(Document(content, metadata))
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """简单关键词匹配检索"""
        if not self.knowledge_base:
            return []
        
        # 简单实现：关键词匹配
        results = []
        query_words = set(query.lower().split())
        
        for doc in self.knowledge_base:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                doc.score = overlap / len(query_words)
                results.append(doc)
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    async def build_context(self, documents: List[Document]) -> str:
        """构建上下文"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc.content}")
        
        return "\n\n".join(context_parts)


# ================== 对话历史管理 ==================
class ConversationHistory:
    """对话历史管理"""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []
    
    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self.messages.append({"role": "user", "content": content})
        self._trim()
    
    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self.messages.append({"role": "assistant", "content": content})
        self._trim()
    
    def _trim(self) -> None:
        """保持历史长度"""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
    
    def get_messages(self, system_prompt: str) -> List[Dict[str, str]]:
        """获取完整消息列表（含system prompt）"""
        return [{"role": "system", "content": system_prompt}] + self.messages
    
    def clear(self) -> None:
        """清空历史"""
        self.messages.clear()


# ================== LLM客户端 ==================
class LLMClient:
    """千问LLM流式客户端"""
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        rag: Optional[RAGInterface] = None,
    ):
        self.config = config or get_config().llm
        self.rag = rag or DummyRAG()
        self.history = ConversationHistory(self.config.max_history_turns)
        
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
    
    def switch_model(self, model_name: str) -> bool:
        """
        切换模型
        
        Args:
            model_name: 模型名称（qwen-flash, qwen-turbo, qwen-plus, qwen-max）
        
        Returns:
            是否切换成功
        """
        return self.config.switch_model(model_name)
    
    def set_rag(self, rag: RAGInterface) -> None:
        """设置RAG实现"""
        self.rag = rag
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.history.clear()
    
    async def chat_stream(
        self,
        user_input: str,
        use_rag: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        流式对话
        
        Args:
            user_input: 用户输入
            use_rag: 是否使用RAG增强
        
        Yields:
            生成的文本片段
        """
        # RAG增强
        if use_rag and get_config().rag.enabled:
            user_input = await self.rag.enhance_query(user_input)
        
        # 添加用户消息到历史
        self.history.add_user_message(user_input)
        
        # 构建消息
        messages = self.history.get_messages(self.config.system_prompt)
        
        try:
            # 流式调用
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=True,
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # 添加助手回复到历史
            self.history.add_assistant_message(full_response)
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            error_msg = "抱歉，我遇到了一些问题，请稍后再试。"
            self.history.add_assistant_message(error_msg)
            yield error_msg
    
    async def chat(self, user_input: str, use_rag: bool = False) -> str:
        """
        非流式对话
        
        Args:
            user_input: 用户输入
            use_rag: 是否使用RAG增强
        
        Returns:
            完整回复
        """
        full_response = ""
        async for chunk in self.chat_stream(user_input, use_rag):
            full_response += chunk
        return full_response


# ================== 句子切分器 ==================
class SentenceSplitter:
    """
    句子切分器
    将LLM流式输出按句子切分，便于TTS合成
    """
    
    # 句子结束标点
    SENTENCE_ENDINGS = r'[。！？.!?]'
    # 需要保留的标点（与后文连接）
    KEEP_WITH_NEXT = r'[，,、：:；;]'
    
    def __init__(self, min_length: int = 5, max_length: int = 100):
        """
        Args:
            min_length: 最小句子长度（过短的句子会合并）
            max_length: 最大句子长度（过长的句子会强制切分）
        """
        self.min_length = min_length
        self.max_length = max_length
        self.buffer = ""
    
    def feed(self, text: str) -> List[str]:
        """
        输入文本，返回完整句子列表
        
        Args:
            text: 新输入的文本片段
        
        Returns:
            完整句子列表（可能为空）
        """
        self.buffer += text
        sentences = []
        
        while True:
            # 查找句子结束标点
            match = re.search(self.SENTENCE_ENDINGS, self.buffer)
            
            if match:
                # 找到句子结束
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                self.buffer = self.buffer[end_pos:].lstrip()
                
                if len(sentence) >= self.min_length:
                    sentences.append(sentence)
                elif sentences:
                    # 短句合并到上一句
                    sentences[-1] += sentence
                else:
                    # 暂存短句
                    self.buffer = sentence + self.buffer
                    break
            elif len(self.buffer) >= self.max_length:
                # 超长句子强制切分
                # 找最后一个逗号或空格
                cut_pos = max(
                    self.buffer.rfind('，', 0, self.max_length),
                    self.buffer.rfind(',', 0, self.max_length),
                    self.buffer.rfind(' ', 0, self.max_length),
                )
                if cut_pos == -1:
                    cut_pos = self.max_length
                
                sentence = self.buffer[:cut_pos].strip()
                self.buffer = self.buffer[cut_pos:].lstrip()
                
                if sentence:
                    sentences.append(sentence)
            else:
                break
        
        return sentences
    
    def flush(self) -> Optional[str]:
        """
        清空缓冲区，返回剩余文本
        
        Returns:
            剩余文本或None
        """
        if self.buffer.strip():
            result = self.buffer.strip()
            self.buffer = ""
            return result
        self.buffer = ""
        return None
    
    def reset(self) -> None:
        """重置缓冲区"""
        self.buffer = ""


# ================== 实时LLM会话 ==================
class RealtimeLLMSession:
    """
    实时LLM会话管理器
    从输入队列读取用户文本，流式生成回复，按句子推送到输出队列
    """
    
    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        config: Optional[LLMConfig] = None,
        rag: Optional[RAGInterface] = None,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.client = LLMClient(config, rag)
        self.splitter = SentenceSplitter()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """启动LLM会话"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("LLM会话已启动")
    
    async def stop(self) -> None:
        """停止LLM会话"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("LLM会话已停止")
    
    def switch_model(self, model_name: str) -> bool:
        """切换模型"""
        return self.client.switch_model(model_name)
    
    def set_rag(self, rag: RAGInterface) -> None:
        """设置RAG实现"""
        self.client.set_rag(rag)
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.client.clear_history()
    
    async def _run(self) -> None:
        """运行处理循环"""
        while self._running:
            try:
                # 等待用户输入
                user_text = await self.input_queue.get()
                
                if user_text is None:  # 结束信号
                    break
                
                logger.info(f"LLM收到输入: {user_text}")
                
                # 重置句子切分器
                self.splitter.reset()
                
                # 流式生成回复
                use_rag = get_config().rag.enabled
                async for chunk in self.client.chat_stream(user_text, use_rag):
                    # 按句子切分
                    sentences = self.splitter.feed(chunk)
                    for sentence in sentences:
                        await self.output_queue.put(sentence)
                        logger.info(f"LLM输出句子: {sentence}")
                
                # 处理剩余文本
                remaining = self.splitter.flush()
                if remaining:
                    await self.output_queue.put(remaining)
                    logger.info(f"LLM输出剩余: {remaining}")
                
                # 发送结束标记
                await self.output_queue.put(None)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"LLM处理出错: {e}")
