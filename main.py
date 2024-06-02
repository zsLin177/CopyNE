import argparse

from asr import CTCAttentionASRParser, CLASCTCAttentionASRParser, CopyNEASRParser, ParaformerASRParser
from supar.utils.logging import init_logger, logger
import os
import torch
from torch.distributed import init_process_group, destroy_process_group
import gradio as gr

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def parse(parser):
    ddp_setup()
    parser.add_argument('--path', help='path to model file')
    parser.add_argument('--pre_model', type=str, default="None")
    parser.add_argument('--seed',
                        '-s',
                        default=1,
                        type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('--num_workers',
                        default=1,
                        type=int)
    parser.add_argument('--e2ener', 
                        action='store_true',
                        help='whether it is an e2ener model')
    parser.add_argument('--char_dict', 
                        default='data/sp_ner/chinese_char.txt', 
                        help='path to the char dict file')
    parser.add_argument('--cmvn', 
                        default='data_to_upload/aishell1_global_cmvn_mel80', 
                        help='global cmvn file')
    parser.add_argument('--config', 
                        default='conf/ctc_mel80.yaml', 
                        help='config file')
    parser.add_argument('--add_bert',
                        action='store_true', 
                        help='whether to add bert')
    parser.add_argument('--bert', 
                        default='bert-base-chinese', 
                        help='which bert model to use')
    parser.add_argument('--frame_length',
                        default=25,
                        type=int)
    parser.add_argument('--frame_shift',
                        default=10,
                        type=int)
    parser.add_argument('--max_frame_num',
                        default=10000,
                        type=int)
    parser.add_argument('--add_context', 
                        action='store_true',
                        help='whether to add context')
    parser.add_argument('--pad_context',
                        default=3,
                        type=float)
    parser.add_argument('--train_ne_dict', default='data/end2end/aishell_train_ner_most-all.vocab')
    parser.add_argument('--dev_ne_dict', default='data/end2end/aishell_dev_ner_random-500.vocab')
    parser.add_argument('--att_type', default='simpleatt', type=str, choices=['contextual', 'crossatt', 'simpleatt'])
    parser.add_argument('--add_copy_loss', action='store_true')
    parser.add_argument('--no_concat', action='store_true')
    parser.add_argument('--use_avg', action='store_true')

    

    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)

    torch.manual_seed(args.seed)
    if int((torch.__version__)[0]) > 1:
        torch.set_float32_matmul_precision('high') # it should be set to high for torch2.0
    init_logger(logger, os.path.join(args.path, f"{args.mode}.log"))
    logger.info('\n' + str(args))

    if args.mode == 'train':
        if not args.add_context:
            parser = CTCAttentionASRParser(args)
        else:
            if not args.add_copy_loss:
                parser = CLASCTCAttentionASRParser(args)
            else:
                parser = CopyNEASRParser(args)
        logger.info(f'{parser.model}\n')
        parser.train()
    elif args.mode == 'evaluate':
        if not args.add_context:
            parser = CTCAttentionASRParser(args)
        else:
            if not args.add_copy_loss:
                parser = CLASCTCAttentionASRParser(args)
            else:
                parser = CopyNEASRParser(args)
        logger.info(f'{parser.model}\n')
        parser.eval()
    elif args.mode == 'api':
        assert args.add_context
        assert args.add_copy_loss
        parser = CopyNEASRParser(args)

        # 定义处理上传的音频和词典文件的函数
        def process_audio(audio_file_path, dictionary_file):
            # 从上传的词典文件中读取词典
            # dictionary = dictionary_file.read().decode("utf-8")
            
            # 调用ASR模型进行转录
            transcription = parser.api(audio_file_path, dictionary_file)
            
            return transcription
        
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
            
            submit_button.click(process_audio, inputs=[audio_input, dictionary_input], outputs=output)

        # 运行Gradio应用
        demo.launch()
        
    destroy_process_group()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--train', default='data/sp_ner/new_train.json', help='path to train file')
    subparser.add_argument('--dev', default='data/end2end/dev_single_bracket.json', help='path to dev file')
    subparser.add_argument('--test', default='data/sp_ner/new_test.json', help='path to test file')

    subparser = subparsers.add_parser('evaluate', help='Evaluation.')
    subparser.add_argument('--input', default='data/aishell1_asr/test.json', help='path to input file')
    subparser.add_argument('--test_ne_dict', default='data/end2end/aishell_dev_ner_allmost300.vocab')
    subparser.add_argument('--res', default='pred.txt', help='path to input file')
    subparser.add_argument('--decode_mode', choices=['attention', 'ctc_greedy_search', 'copy_attention'], help='decoding mode to use')
    subparser.add_argument('--beam_size', default=10, type=int, help='beam size')
    subparser.add_argument('--copy_threshold', default=0.9, type=float, help='threshold for copying')

    subparser = subparsers.add_parser('api', help='API')
    subparser.add_argument('--test_ne_dict', default='None')
    subparser.add_argument('--beam_size', default=10, type=int, help='beam size')

    parse(parser)
    
