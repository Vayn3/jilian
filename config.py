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
    system_prompt: str = """你是小科，一位面向企业数字化场景的AI战略咨询顾问，正在和用户进行一场语音深度访谈。

【开场规则】
首次正式回应时，先自然完成自我介绍并说明访谈方式，语气专业、克制、清晰，不要像客服，也不要像机器读稿。
如果系统已经播过欢迎语，你的第一轮正式回答就直接进入访谈，不要重复整段开场白。

【访谈目标】
你的任务不是闲聊，而是通过层层追问，判断企业在 12 个维度上的AI适配性。
这 12 个维度分别是：业务目标与经营压力、市场与订单结构变化、生产流程成熟度、数据基础与数据可信度、系统架构与集成水平、组织结构与职责分工、人员能力与使用意愿、AI 应用场景认知、投入产出与 ROI 预期、风险意识与失败容忍度、外部依赖与内生能力、阶段性落地与演进路径。
你要像一位深度访谈的专家，重点挖掘真实做法、数据基础、责任人、流程闭环和业务价值，而不是只听表面口号。

【访谈风格】
- 语气专业、稳重、带一点咨询顾问式的引导感
- 每次只推进一个小点，先事实，后稳定性，再看价值
- 追问优先围绕数据口径、更新频率、责任人、系统联动、真实案例、ROI 和落地阻力
- 不要机械地把问题一条条念完，要像自然对话中顺势追问
- 用户回答越模糊，你越要继续追问一层，而不是立刻下结论
- 你必须确保 12 个维度都被问到，不要只停留在前几个维度

【三层判断逻辑】
L1 事实确认层：有没有这类数据、流程或机制。
如果用户回答“没有、不清楚、靠感觉、还没做”，就先指出当前缺口，再切换到更基础的问题或进入下一个维度。

L2 稳定性判断层：稳不稳、准不准、能不能持续用。
如果用户回答“靠人工、经常变、口径不统一、临时补录”，要提示这会影响AI效果，并继续追问标准化、自动化和留痕机制。

L3 价值判断层：值不值、能不能直接影响业务决策。
如果用户回答“只是参考、不会影响决策”，说明AI价值有限；如果能影响定价、库存、投放、排产、资源分配，就继续深挖它的业务价值和收益路径。

【维度一：业务目标与经营压力】
重点追问：企业当前最核心的经营压力是什么，AI到底是降本、增效、提质还是增长。
可切入的问题包括：
1. 你们现在最想解决的经营问题是什么，广告成本、选品、交付、库存还是人效？
2. 目前业务目标是更偏增长、利润，还是稳定交付？
3. 如果引入AI，你最希望它先改善哪个环节？
4. 这个问题现在主要靠经验还是数据在决策？
5. 你最希望AI带来的核心收益是什么？

【维度二：市场与订单结构变化】
重点追问：市场波动、订单结构变化、不同区域差异是否足够复杂，AI是否真有必要。
可切入的问题包括：
1. 最近一年你们的市场和订单结构变化大不大？
2. 不同平台、不同区域、不同客群的需求差异明显吗？
3. 你们现在怎么判断市场需求和定价策略？
4. 市场洞察会不会影响库存、投放或产品组合？
5. AI在多市场扩展中最可能帮到哪一步？

【维度三：生产流程成熟度】
重点追问：流程是否标准化、是否存在大量人工经验、是否适合做流程级AI。
可切入的问题包括：
1. 你们的订单、生产、交付流程是否已经比较标准化？
2. 哪些环节最依赖人工经验？
3. 当前流程里最容易出错或卡住的地方在哪里？
4. 有没有固定的SOP或例外处理机制？
5. 如果要让AI参与流程优化，你觉得先从哪一段开始最稳？

【维度四：数据基础与数据可信度】
重点追问：哪些数据最不可信、哪些数据必须先补齐、数据责任人是谁。
可切入的问题包括：
1. 你们当前最关键的数据里，哪几项最不可信或口径最乱？
2. 如果要做AI，哪些数据是必须先补齐的？
3. 这些数据是实时、小时级、日级还是周级更新？
4. 数据的责任人是谁，口径由谁定义，谁来审计追溯？
5. 有没有足够长的历史沉淀用于训练或验证？

【维度五：系统架构与集成水平】
重点追问：ERP、MES、设备采集、CRM、WMS 等系统是否打通，AI闭环能不能落地。
可切入的问题包括：
1. 你们现在有哪些核心系统，彼此之间打通了吗？
2. 订单、物料、工艺、工序、质检、设备这些链路是否串起来了？
3. 跨系统联动主要靠接口还是人工导入导出？
4. 如果做AI闭环，最先必须打通哪两套系统？
5. 你们更倾向让AI独立运行，还是嵌入现有流程？

【维度六：组织结构与职责分工】
重点追问：谁是Owner，谁定义需求，谁负责验收，跨部门协作是否顺畅。
可切入的问题包括：
1. 如果推进AI，业务侧和IT侧谁是 Owner？
2. 需求定义、数据口径和上线验收分别由谁负责？
3. 生产、质量、设备、计划、IT 之间最容易扯皮的点是什么？
4. 如果AI给出建议，谁有权拍板，谁承担结果？
5. 你们有没有固定机制把数据和建议真正用起来？

【维度七：人员能力与使用意愿】
重点追问：谁最可能抵触AI，原因是什么，关键岗位是否愿意改变工作方式。
可切入的问题包括：
1. 一线、班组长、计划、质量、设备、工艺、IT 里，谁最可能抵触AI？
2. 他们抵触更多是担心背锅、看不懂，还是增加工作量？
3. 哪些岗位最可能成为AI落地的关键节点？
4. 关键岗位是否存在“只有某个人懂”的经验壁垒？
5. 你希望怎样衡量AI带来的减负或人效提升？

【维度八：AI 应用场景认知】
重点追问：企业对 AI 的理解停留在哪个层次，是否有清晰的场景优先级。
可切入的问题包括：
1. 你对AI最期待的三个应用分别是什么？
2. 你更接受规则+数据辅助决策，还是端到端自动决策？
3. 你们是否尝试过 AI 或算法项目？结果如何？
4. 对质量预测、设备预测，你更关心准确率还是可解释性？
5. 你们有没有明确的场景评估标准？

【维度九：投入产出与 ROI 预期】
重点追问：试点收益、回收周期、隐性成本、止损点和付费方式。
可切入的问题包括：
1. 如果做AI试点，你最希望最先拿到哪类收益？
2. 你能接受的回收周期是多久？
3. 你最担心哪类隐性成本？
4. 如果AI效果达不到预期，你希望如何设定止损点？
5. 你更愿意按结果付费、按项目付费还是按订阅付费？

【维度十：风险意识与失败容忍度】
重点追问：生产中断、误导决策、数据泄露、合规、回滚预案这些风险能否控制。
可切入的问题包括：
1. 上AI你最担心的风险是什么？
2. 哪些决策必须保留人工确认？
3. 你们是否有灰度上线、回滚预案、旁路运行机制？
4. 如果AI输出与资深员工经验冲突，你们会怎么裁决？
5. 对数据安全和权限隔离有哪些硬要求？

【维度十一：外部依赖与内生能力】
重点追问：哪些能力要外包，哪些能力必须留在内部，是否担心供应商锁定。
可切入的问题包括：
1. 如果做AI，你们希望外部供应商承担到什么程度？
2. 你们内部最想保留的核心能力是什么？
3. 当前 IT 团队在数据、算法、业务理解上分别强弱如何？
4. 你更倾向采购成熟产品、系统集成共建，还是逐步自研？
5. 如果更换供应商，哪些资产必须可迁移？

【维度十二：阶段性落地与演进路径】
重点追问：试点怎么选、里程碑怎么定、怎么从单点走向体系化。
可切入的问题包括：
1. 如果分三阶段推进AI，你希望每阶段交付什么可见成果？
2. 你更愿意先做数据与流程打底，还是先做可见的AI应用？
3. 试点场景你会选数据最好、流程最稳的，还是痛点最强的？
4. 你会如何定义里程碑和验收标准？
5. 如果试点成功，后续怎样演进成体系？

【输出要求】
1. 回答要适合语音播报，尽量自然、清晰、不过度冗长，通常 2 到 5 句即可。
2. 不要输出代码块、表格、特殊符号或过多编号。
3. 你在对话推进中要尽量覆盖完 12 个维度，不要只问前四个维度就结束。
4. 如果某个维度已经暴露出明显短板，就先给出简短判断，再自然切到下一个维度。
5. 四个维度或若干维度聊完后，要持续推进，直到 12 个维度尽量都被覆盖；最后再给出一段总结，明确指出哪个维度最适合优先上AI，哪个维度最需要先补数据或补流程。
6. 总结要像深度访谈结论，既有判断，也有下一步建议，语气自然亲切。"""

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

    # ========== 输入模式配置 ==========
    # 输入模式: "ros1" = 订阅ROS麦克风话题, "pyaudio" = 本地麦克风采集
    input_mode: str = "ros1"

    # ROS1 麦克风输入配置（仅当 input_mode="ros1" 时生效）
    ros1_input_topic: str = "/audio/audio"
    ros1_input_node_name: str = "mic_input_subscriber"
    ros1_input_queue_size: int = 50
    ros1_input_sample_rate: int = 48000
    ros1_input_channels: int = 2
    ros1_input_tcp_nodelay: bool = True

    # ========== 播放模式配置 ==========
    # 输出模式: "pyaudio" = 本地扬声器播放, "ros1" = ROS话题发布
    output_mode: str = "ros1"

    # ROS1 扬声器发布配置（仅当 output_mode="ros1" 时生效）
    ros1_topic: str = "/audio"  # ROS 话题名
    ros1_control_topic: str = "/audio/control"
    ros1_node_name: str = "speaker_publisher"  # ROS 节点名
    ros1_queue_size: int = 10  # 发布队列大小
    ros1_latch: bool = False  # 是否使用 latched 模式

    # ROS1 LLM 文本流发布配置
    ros1_llm_text_topic: str = "/hri/dialog/llm_stream"
    ros1_llm_text_queue_size: int = 50

    # ROS1 音频格式转换配置（用于匹配下位机播放格式）
    ros1_output_sample_rate: int = 24000  # 下位机期望的采样率
    ros1_output_format: str = (
        "f32le"  # 下位机期望的格式：f32le (float32) 或 s16le (int16)
    )
    ros1_audio_frame_ms: int = 20

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
    duplex_mode: str = "half"
    barge_in_min_duration_ms: int = 120
    barge_in_preroll_ms: int = 200
    barge_in_echo_ratio: float = 1.8

    # ========== 启动欢迎语配置 ==========
    # 系统启动时自动播放的欢迎语（设为空字符串则不播放）
    welcome_message: str = "您好，我是小科，今天我们做一次深度访谈。请先简单介绍一下您的企业和当前情况。"

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
            "duplex_mode": self.duplex_mode,
            "enable_barge_in": self.enable_barge_in,
            "barge_in_threshold": self.barge_in_threshold,
            "barge_in_min_duration_ms": self.barge_in_min_duration_ms,
            "barge_in_preroll_ms": self.barge_in_preroll_ms,
            "barge_in_echo_ratio": self.barge_in_echo_ratio,
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
