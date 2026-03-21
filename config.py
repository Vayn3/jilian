# -*- coding: utf-8 -*-
"""
级联式武汉话实时语音对话系统 - 全局配置
支持：豆包ASR → 千问LLM → 讯飞TTS（武汉话）
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ASRConfig:
    """豆包ASR配置"""

    # API认证
    app_key: str = "7381194560"
    access_key: str = "PmMJqNvQDStP4xpTi4pnuO83F793BplS"

    # WebSocket URL
    # bigmodel: 双向流式（每包立即返回）
    # bigmodel_nostream: 流式输入（发完再返回）
    # bigmodel_async: 双向流式优化版（推荐，只有结果变化时返回）
    url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"

    # 音频配置
    sample_rate: int = 16000
    bits: int = 16
    channels: int = 1  # 单声道

    # VAD与端点检测配置
    end_window_size_ms: int = 200  # 静音判停时间（ms），越小越快但可能切句过早
    vad_silence_threshold: int = 400  # VAD能量阈值
    max_silence_ms: int = 200  # 最大静音时长（ms）
    max_record_ms: int = 15000  # 单次最大录音时长（ms）

    # 分片配置
    segment_duration_ms: int = 200  # 每包音频时长（ms）

    # 功能开关
    enable_itn: bool = True  # 数字规范化
    enable_punc: bool = True  # 标点符号
    enable_ddc: bool = False  # 顺滑（关闭可降低延迟）
    show_utterances: bool = True  # 输出分句信息


@dataclass
class LLMConfig:
    """千问LLM配置"""

    # API认证
    api_key: str = "sk-9d5d8ee616b740cd9e58a1152f84f471"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 模型选择: qwen-flash（快速）或 qwen-plus（准确）
    model: str = "qwen-flash"  # 默认使用flash，延迟更低
    available_models: tuple = ("qwen-flash", "qwen-turbo", "qwen-plus", "qwen-max")

    # 生成参数
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # 流式输出
    stream: bool = True

    # System Prompt
    system_prompt: str = """你是一个友好的AI助手，你叫小科，正在和用户进行语音对话。
【角色设定】 你是一位高级AI战略咨询顾问。你的任务是通过语音访谈，评估企业在“市场、客户、产品、趋势”四个维度的AI适配性。你的风格专业、犀利且具有引导性，能够通过追问挖掘企业真实的数据基础。

【核心指令：随机抽题与级联判定】 每一个维度我为你准备了6个杀手锏母题（源自母题库）。在访谈每个维度时，请随机抽取其中4个进行提问，但必须遵循“从基础（L1）到细节（L2）再到价值（L3）”的自然对话流。

【访谈逻辑：三层级联过滤器】 对于用户的每一个回答，你必须在后台进行实时逻辑匹配（参考）：

L1 事实确认层（有没有）：

判定： 用户说“没有/不清楚/靠感觉” -> 立即停止该维度深挖，结论为“补基础”。

判定： 用户说“有/系统里有” -> 继续。

L2 稳定性判断层（稳不稳）：

判定： 用户说“靠人盯/经常变/忙就不做” -> 判定不可直接上AI，结论为“需做数据治理”。

判定： 用户说“系统自动/SOP/统一口径” -> 继续。

L3 决策价值层（值不值）：

判定： 用户说“只是参考/不影响决策” -> 结论为“AI仅为装饰”。

判定： 用户说“影响定价/改变库存/决定投放” -> 结论为“值得上AI（高ROI）”。

【维度一：市场数据与销售预测】（题库池：6选4） 请根据对话流，从以下随机选4个提问：


[L1] 贵公司是否建立了具体的市场数据指标体系来分析趋势？ (Q2)


[L1/L2] 请描述贵公司目前是如何收集市场数据以支撑销售预测的？ (Q1)


[L2] 贵公司是否有自动化工具对市场趋势进行持续监控，还是主要靠人工？ (Q33)


[L2] 你们如何确保市场数据的更新频率能满足AI系统的实时性要求？ (Q10)


[L3] 企业是否具备将市场洞察直接转化为实际操作策略（如库存或定价）的能力？ (Q20)


[L3] 你们是否通过ROI指标来决定市场分析类AI应用的优先级？ (Q43)

【维度二：客户行为与精准营销】（题库池：6选4） 请根据对话流，从以下随机选4个提问：


[L1] 贵公司是否建立了客户数据采集体系，数据中是否包含购买行为和偏好？ (Q61/Q62)


[L2] 这些客户数据是实时更新以反映最新行为，还是定期批量处理？ (Q63)


[L2] 你们是否对采集的客户数据进行了标准化的清洗和预处理？ (Q64)


[L2] 是否使用自动化工具来监控客户行为，还是依赖人工报表？ (Q92)


[L3] 企业是否能真正将客户洞察转化为具体的营销执行方案（如自动投放）？ (Q80)


[L3] 你们是否通过ROI来判断客户数据分析项目的优先级？ (Q107)

【维度三：产品性能与创新优化】（题库池：6选4） 请根据对话流，从以下随机选4个提问：


[L1] 你们目前是否有产品性能数据用于后续的分析或AI模型训练？ (Q121)


[L1/L2] 产品数据的采集流程是标准化的吗？便于AI直接使用吗？ (Q123)


[L2] 是否有自动化工具专门采集产品性能或销售数据？ (Q152)


[L2] 企业是否监控产品数据的异常趋势？是通过系统报警还是人工查看？ (Q128)


[L3] 团队是否能快速将数据分析结果应用到实际的产品策略调整中？ (Q138)


[L3] 是否通过ROI指标来决定优先优化哪些产品线？ (Q164)

【维度四：行业趋势与战略决策】（题库池：6选4） 请根据对话流，从以下随机选4个提问：


[L1] 企业是否专门收集行业趋势数据（技术、市场、客户）用于辅助决策？ (Q181/Q182)


[L2] 趋势数据的采集是标准化且可自动更新的吗？ (Q183)


[L2] 是否使用自动化工具跟踪行业趋势，还是依赖外部报告？ (Q212)


[L2] 这些数据是否完整且足够及时，以支持战略层面的决策？ (Q187)


[L3] 团队能否将趋势分析结果直接转化为具体的业务策略？ (Q193)


[L3] ROI分析是否被用于判断趋势分析项目的优先级？ (Q225)

【输出要求】

自然对话： 不要机械地读题，用交谈的语气抛出问题。例如：“说到产品方面，我很好奇你们现在有专门的性能数据采集流程吗？”

即时反馈： 如果用户在L1或L2阶段触发了“停止词”（如：没有、靠人工），请礼貌地指出风险（如：“嗯，如果主要是靠人工，数据的一致性可能会影响AI的效果”），然后跳过该维度剩余问题，直接进入下一个维度。

最终诊断： 四个维度聊完后，请结合用户的回答，生成一段总结。指出哪个维度是“准备好上AI的（高价值）”，哪个维度“基础薄弱（需补课）”。
请注意：
1. 回答要简洁明了，适合语音播报
2. 避免使用特殊符号、表情、代码块等不适合语音的内容
3. 回答长度适中，一般不超过5句话
4. 语气自然亲切"""

    # 对话历史
    max_history_turns: int = 10  # 最大保留对话轮数

    def switch_model(self, model_name: str) -> bool:
        """切换模型"""
        if model_name in self.available_models:
            self.model = model_name
            return True
        return False


@dataclass
class TTSConfig:
    """讯飞TTS配置"""

    # API认证
    app_id: str = "6130dc73"
    api_key: str = "5af3f5aea48cb34ed691efee2a18780f"
    api_secret: str = "OGM2ZGZmNTI4OTJjZjgyNjM4ZThjOTk0"

    # WebSocket URL
    url: str = "wss://tts-api.xfyun.cn/v2/tts"

    # 音频配置
    aue: str = "raw"  # raw=PCM, lame=MP3
    auf: str = "audio/L16;rate=16000"  # 采样率
    sample_rate: int = 16000

    # 发音人配置 - 武汉话
    vcn: str = "x4_yezi"  # "x2_xiaowang"  # 武汉话发音人

    # 语音参数
    speed: int = 50  # 语速 0-100，50为正常
    volume: int = 50  # 音量 0-100
    pitch: int = 50  # 音高 0-100

    # 文本编码
    tte: str = "utf8"  # utf8 或 unicode(utf16le)


@dataclass
class AudioConfig:
    """音频设备配置"""

    # 采样参数（麦克风硬件采样率）
    sample_rate: int = 48000  # 麦克风采样率（硬件支持）
    sample_width: int = 2  # 16bit = 2 bytes
    channels: int = 1  # 单声道

    # 设备索引（None表示使用默认设备）
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None

    # 缓冲区配置
    input_chunk_ms: int = 20  # 输入帧长度（ms）
    output_buffer_size: int = 1024  # 输出缓冲区大小

    # 回声消除配置
    enable_aec: bool = True  # 启用回声消除
    aec_frame_size: int = 160  # AEC帧大小（采样点数，10ms@16kHz）
    aec_filter_length: int = 2048  # AEC滤波器长度

    # 噪声抑制
    enable_ns: bool = True  # 启用噪声抑制
    ns_level: int = 2  # 噪声抑制级别 0-3

    # 自动增益控制
    enable_agc: bool = True  # 启用自动增益
    agc_target_level: int = 3  # AGC目标电平 0-31

    # ========== 播放模式配置 ==========
    # 输出模式: "pyaudio" = 本地扬声器播放, "ros1" = ROS话题发布
    output_mode: str = "ros1"

    # ROS1 扬声器发布配置（仅当 output_mode="ros1" 时生效）
    ros1_topic: str = "/audio"  # ROS 话题名
    ros1_node_name: str = "speaker_publisher"  # ROS 节点名
    ros1_queue_size: int = 10  # 发布队列大小
    ros1_latch: bool = False  # 是否使用 latched 模式

    # ROS1 LLM 文本流发布配置
    ros1_llm_text_topic: str = "/dialog/llm_stream"
    ros1_llm_text_queue_size: int = 50

    # ROS1 音频格式转换配置（用于匹配下位机播放格式）
    ros1_output_sample_rate: int = 24000  # 下位机期望的采样率
    ros1_output_format: str = (
        "f32le"  # 下位机期望的格式：f32le (float32) 或 s16le (int16)
    )

    @property
    def bytes_per_frame(self) -> int:
        """每帧字节数"""
        return int(
            self.sample_rate
            * self.sample_width
            * self.channels
            * self.input_chunk_ms
            / 1000
        )

    @property
    def frames_per_second(self) -> int:
        """每秒帧数"""
        return 1000 // self.input_chunk_ms


@dataclass
class RAGConfig:
    """RAG检索增强配置（预留接口）"""

    enabled: bool = False

    # 向量数据库配置
    vector_db_url: str = ""
    vector_db_api_key: str = ""

    # 检索配置
    top_k: int = 3  # 检索文档数量
    similarity_threshold: float = 0.7  # 相似度阈值

    # 嵌入模型
    embedding_model: str = "text-embedding-v2"  # 百炼嵌入模型


@dataclass
class SystemConfig:
    """系统总配置"""

    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # 打断配置
    enable_barge_in: bool = True  # 启用打断功能
    barge_in_threshold: int = 600  # 打断检测能量阈值

    # ========== 启动欢迎语配置 ==========
    # 系统启动时自动播放的欢迎语（设为空字符串则不播放）
    welcome_message: str = "大家好，我是华科机器人小科，很高兴见到你！"

    # ========== UDP 动作控制配置 ==========
    # 语音关键词 UDP 通道（发送动作指令给下位机）
    voice_udp_host: str = "127.0.0.1"
    voice_udp_port: int = 5557

    # MIC 指令 UDP 通道（发送收/递麦克风指令）
    mic_udp_host: str = "127.0.0.1"
    mic_udp_port: int = 5558

    # 关键词检测开关
    enable_keyword_detection: bool = True  # 启用关键词检测

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "asr": self.asr.__dict__,
            "llm": {
                k: v for k, v in self.llm.__dict__.items() if k != "available_models"
            },
            "tts": self.tts.__dict__,
            "audio": self.audio.__dict__,
            "rag": self.rag.__dict__,
        }


# 全局配置实例
config = SystemConfig()


def get_config() -> SystemConfig:
    """获取全局配置"""
    return config


def update_config(**kwargs) -> SystemConfig:
    """更新配置"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
