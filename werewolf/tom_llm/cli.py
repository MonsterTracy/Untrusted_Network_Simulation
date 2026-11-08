"""唯一的 uns tom-llm 命令入口。"""


def register(commands):
    root = commands.add_parser('tom-llm',help='集合 Baseline：准备、顺序运行、安全恢复和统一评价')
    root.add_argument('--config',default='tom-llm.yaml')
    sub = root.add_subparsers(dest='operation',required=True)
    for name in ('prepare','run','status','stop'):
        p = sub.add_parser(name)
        p.add_argument('--run-id',required=True)
        if name in ('prepare','run'):
            p.add_argument('--model',required=True,choices=['qwen3','yi','mistral'])
        if name in ('run','stop'):
            p.add_argument('--execute-live',action='store_true')
        if name=='run':
            p.add_argument('--resume',action='store_true')
    p = sub.add_parser('report')
    p.add_argument('--sealed-run',action='append',required=True,help='model=Run目录，恰好三个模型')
    p.add_argument('--sealed-run-digest',action='append',required=True,help='model=可信结果Seal摘要')
    p.add_argument('--output',required=True)
