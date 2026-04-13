# 全双工语音与 ROS 可中断播放改造计划

## Summary
- 目标是把系统从“边播边停录”的半双工，改成“持续收音、用户可随时打断、下位机扬声器可立即止播”的全双工，同时保留现有半双工作为默认启动行为。
- 默认策略定为：`--duplex-mode half|full`，默认 `half`；只有显式传 `full` 才启用全双工链路，降低回归风险。
- ROS 麦克风输入会增加一跳延迟，主要来自发布包长、序列化、ROS 队列和重采样；方案默认按 `10-20ms` 音频包、较小 `queue_size`、`tcp_nodelay=True` 来设计，把新增延迟控制在“几十毫秒”量级，避免变成“几百毫秒”。

## Public APIs / Interfaces
- 新增启动参数：`--duplex-mode {half,full}`，默认 `half`。
- 保留 `--no-barge-in` 作为兼容别名，效果等同于强制 `half`，新代码里以 `duplex_mode` 为主。
- 新增配置：
  - `SystemConfig.duplex_mode = "half"`
  - `SystemConfig.barge_in_min_duration_ms = 120`
  - `SystemConfig.barge_in_preroll_ms = 200`
  - `SystemConfig.barge_in_echo_ratio = 1.8`
  - `AudioConfig.ros1_control_topic = "/audio/control"`
  - `AudioConfig.ros1_audio_frame_ms = 20`
  - `AudioConfig.ros1_input_tcp_nodelay = True`
- 统一用现有 `utterance_id` 作为整条回复链路的代次号；不要再引入第二套 `playback_id`。
- 内部队列从裸 `str/bytes/None` 改成带代次的结构体：
  - `LLMSentence(utterance_id, text, is_final)`
  - `TTSChunk(utterance_id, chunk_seq, audio_bytes, is_last)`
- ROS 控制面新增一个 `std_msgs/String` JSON topic：`/audio/control`
  - 消息格式固定为：`{"cmd":"stop","utterance_id":N,"reason":"barge_in","timestamp":...}`
- ROS 音频数据面在 `full` 模式下改成“带固定头的二进制帧”，仍复用 `audio_common_msgs/AudioData` 或 `ByteMultiArray.data`
  - 头部固定字段：`magic + version + flags + utterance_id + chunk_seq + payload_len`
  - 负载仍是原 PCM 数据
- 下位机继续兼容旧的“纯裸音频包”；自动识别新帧头和旧格式，保证半双工旧链路不被破坏。

## Implementation Changes
- 中央协调层改成“统一分配 utterance_id”。
  - `main.py` 在每次 ASR 最终结果进入新一轮机器人回复时生成 `utterance_id`，同时传给 LLM 文本流、TTS、ROS 音频和 ROS 文本流。
  - `ros_dialog.py` 不再自己递增 `utterance_id`，而是接收外部传入的当前代次。
- 新增一个原子中断入口 `interrupt_current_utterance(reason)`。
  - 取消当前 LLM 流式生成
  - 停止当前 TTS 合成
  - 清空 `llm_queue` 和 `tts_queue` 中旧代次数据
  - 如果 `output_mode=ros1`，立刻向 `/audio/control` 发 `stop`
  - 将当前 `utterance_id` 标记为失效，晚到的旧 chunk 一律丢弃
- `RealtimeLLMSession` 增加中断能力。
  - 活跃回复改成单独 task，可在打断时直接 cancel
  - `SentenceSplitter` 在打断时 reset
  - 只有 `utterance_id` 仍是当前代次时，才允许往下游写句子
- `RealtimeTTSSession` 增加中断与代次过滤。
  - 输入改收 `LLMSentence`
  - 打断后立即停止当前合成，不再让旧句子继续排队
  - `on_tts_end` 只在“当前代次且未被打断”时触发
- `RealtimeAudioPlaySession` 增加代次过滤和真实 flush。
  - 本地 `pyaudio` 输出时，打断后清空待播队列
  - ROS 输出时，发控制消息并停止继续发布旧代次 chunk
- 把当前 `VAD loop` 改成真正的全双工输入处理。
  - 收音始终开启，`half` 只是“说话时不允许打断”
  - `full` 时在 `THINKING/SPEAKING` 期间也持续检测用户语音
  - 删除当前打断后立刻 `resume()` 的逻辑
  - 删除 `DialogManager.handle_barge_in()` 里的固定 `0.1s` 睡眠，改为立刻切回 `LISTENING`
- 打断检测改成“固定阈值 + 自适应回声底噪”。
  - 两种输入模式都先做现有 AEC/NS
  - 只有连续 `N` 帧超过 `max(static_threshold, playback_leak_floor * echo_ratio)` 才算打断
  - 加一个 `200ms` pre-roll ring buffer，避免用户开口前几个音节被阈值触发过程吃掉
- ROS 麦克风输入链路统一成固定帧长。
  - 订阅后先做声道归一、采样率归一
  - 若单条 ROS 消息过大，拆成 `20ms` 子帧再送 VAD/打断检测
  - 订阅端打开 `tcp_nodelay=True`，并把 `queue_size` 维持在小值
- `ros_audio.py` 增加全双工发布协议。
  - `half` 模式保持原发布方式
  - `full` 模式给每个音频 chunk 加头部并带上 `utterance_id/chunk_seq`
  - 中断和退出时都发 `stop`
- `robot/ros_audio_player.py` 增加下位机控制面和硬 flush。
  - 新增 `/audio/control` 订阅
  - 收到 `stop` 后清空内部队列、把该 `utterance_id` 标记为 canceled、发布 `status=false`
  - 为了尽快把扬声器停掉，执行 `stop_stream() -> close() -> reopen()`，而不是等旧 buffer 自然放完
  - 音频回调只接受“当前有效代次”的 chunk；任何被取消或过期的 chunk 直接丢弃

## Test Plan
- 单元测试
  - `utterance_id` 代次过滤：旧 LLM 句子、旧 TTS chunk、旧 ROS 音频包不会复活
  - `interrupt_current_utterance()` 后两个队列都被正确清空
  - ROS 音频帧头打包/解包正确
  - 下位机收到 `stop` 后会丢弃晚到的旧 chunk
- 集成测试
  - `pyaudio input + pyaudio output`：`half/full` 都跑通
  - `pyaudio input + ros1 output`：说话打断后，下位机扬声器立即停播
  - `ros1 input + ros1 output`：用户在机器人讲话时插话，系统进入 LISTENING，旧回复不会继续播放
  - 在 `THINKING` 阶段打断和在 `SPEAKING` 阶段打断都能正确取消旧轮次
- 回归测试
  - 默认不传参数时仍是现有半双工行为
  - 旧的裸音频发布/订阅链路仍可工作
  - `/audio_playing_status` 仍能正确反映播放开始/结束
- 延迟测试
  - ROS 麦克风发布包长分别测 `10/20/50/100ms`
  - 记录“用户开口到下位机停播”的时间
  - 结论应明确写入文档：`50ms+` 包长会明显拖慢全双工打断

## Assumptions / Defaults
- 可以同时修改上位机代码和下位机 `robot/ros_audio_player.py`。
- 允许新增 ROS 控制 topic，不走 UDP 停播链路。
- 启动参数只要求切换半双工/全双工；`input_mode=ros1|pyaudio` 继续沿用现有配置机制。
- ROS 麦克风全双工的体验上限取决于外部麦克风 publisher 的包长；如果外部设备只能发大包，功能仍可用，但打断响应会明显变钝。
