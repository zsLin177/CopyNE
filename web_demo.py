import gradio as gr
import tempfile

# 假设有一个现成的 ASR 模型函数
def transcribe(audio_file_path, dictionary):
    # 这里是调用您的ASR模型的伪代码
    # 例如，加载音频文件，使用模型进行转录
    # result = asr_model.transcribe(audio_file_path, dictionary)
    result = "这是一个伪造的转录结果。"  # 请替换为实际的模型调用
    return result

# 定义处理上传的音频和词典文件的函数
def process_audio(parser, audio_file_path, dictionary_file):
    # 从上传的词典文件中读取词典
    # dictionary = dictionary_file.read().decode("utf-8")
    
    # 调用ASR模型进行转录
    transcription = parser.api(audio_file_path, dictionary_file)
    
    return transcription

def create_demo(parser):
    # 创建Gradio界面
    with gr.Blocks() as demo:
        # 使用HTML和CSS添加动态效果
        gr.HTML("""
        <style>
        @keyframes rainbow {
            0% { color: red; }
            14% { color: orange; }
            28% { color: yellow; }
            42% { color: green; }
            57% { color: blue; }
            71% { color: indigo; }
            85% { color: violet; }
            100% { color: red; }
        }
        .rainbow-text {
            animation: rainbow 5s linear infinite;
        }
        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }
        .typing-demo {
            display: inline-block;
            overflow: hidden;
            white-space: nowrap;
            font-size: 1.5em;
            animation: typing 4s steps(40, end), blink-caret .75s step-end infinite;
        }
        @keyframes blink-caret {
            from, to { border-color: transparent; }
            50% { border-color: orange; }
        }
        .typing-finished {
            animation-fill-mode: forwards;
        }
        </style>
        <div style="text-align: center;">
            <h1 class="typing-demo"><span class="rainbow-text">CopyNE Demo</span></h1>
            <h3>https://github.com/zsLin177/CopyNE</h3>
        </div>
        <script>
            setTimeout(() => {
                document.querySelector('.typing-demo').classList.add('typing-finished');
            }, 4000);

            // 添加打字效果到output
            let outputText = document.querySelector('.output_textbox');
            let outputContent = outputText.innerHTML;
            outputText.innerHTML = '';
            let i = 0;
            let typingInterval = setInterval(() => {
                if (i < outputContent.length) {
                    outputText.innerHTML += outputContent.charAt(i);
                    i++;
                } else {
                    clearInterval(typingInterval);
                }
            }, 50);
        </script>
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 上传音频文件或使用麦克风录制")
                audio_input = gr.Audio(type="filepath", label="音频文件")
                gr.Markdown("#### 上传用户词典文件")
                dictionary_input = gr.File(label="词典文件")
                submit_button = gr.Button("开始转录")
            
            with gr.Column():
                gr.Markdown("#### 转录结果")
                output = gr.Textbox(label="", placeholder="转录结果将显示在这里", lines=10)
        
        submit_button.click(process_audio, inputs=[parser, audio_input, dictionary_input], outputs=output)

    # 运行Gradio应用
    demo.launch()
