from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import re

# 推荐将API Key和Base URL放在环境变量中
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")

if not QWEN_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY，获取你的Qwen大模型API密钥。")

llm = ChatOpenAI(
    openai_api_key=QWEN_API_KEY,
    openai_api_base=QWEN_API_BASE,
    model=QWEN_MODEL
)

# 优化的问答对提取Prompt
QA_EXTRACTION_PROMPT = """你是一个专业的文档问答对提取专家。你的任务是从给定的文本中准确提取问答对。

## 任务说明
- 识别文本中的问答结构，通常以"问"开头，"答"开头
- 正确区分真正的问答对和文本中可能包含的"问"、"答"字眼
- 确保每个问答对都是完整且有意义的
- 保持原始文本的准确性和完整性

## 提取规则
1. 问答对必须以"问"开头，以"答"开头
2. 问题部分从"问"字后开始，到下一个"答"字前结束
3. 答案部分从"答"字后开始，到下一个"问"字前结束（如果有的话）
4. 如果答案中包含"问"字，不要将其误认为新的问题开始
5. 去除问答对前后的标点符号和多余空格

## 输出格式
请按照以下格式输出问答对：
问：[问题内容]
答：[答案内容]

---

## 示例1
输入文本：
问您把事情的经过如实陈述一下答我是今天早上也就是二零二二年六月十四日六七点钟从叉叉市家中乘坐高铁前往武汉 然后在下午十四时许从武汉市乘坐高铁到达了杭州东站我这次来杭州就是为了前往我的前雇主阿雅家中索要薪资的大概在十七时许的样子我到了杭州市叉叉区钻石城二幢楼下

输出结果：
问：您把事情的经过如实陈述一下
答：我是今天早上也就是二零二二年六月十四日六七点钟从叉叉市家中乘坐高铁前往武汉 然后在下午十四时许从武汉市乘坐高铁到达了杭州东站我这次来杭州就是为了前往我的前雇主阿雅家中索要薪资的大概在十七时许的样子我到了杭州市叉叉区钻石城二幢楼下

## 示例2
输入文本：
问您当时为什么要这样做答因为我没有钱吃饭住宿所以很着急问您知道这样做是违法的吗答我知道不对但是当时太冲动了

输出结果：
问：您当时为什么要这样做
答：因为我没有钱吃饭住宿所以很着急
问：您知道这样做是违法的吗
答：我知道不对但是当时太冲动了

---

现在请处理以下文本：

{text}

请严格按照上述格式输出问答对："""

PROMPT_TEMPLATES = {
    "qa": QA_EXTRACTION_PROMPT,
    "summary": "请对以下文本进行简要总结：\n{text}",
    "keywords": "请提取以下文本的关键词：\n{text}",
}

def process_text_with_qwen(text, prompt_type="qa"):
    """
    用Qwen大模型处理文本。
    :param text: 输入文本
    :param prompt_type: 处理类型（qa/summary/keywords等）
    :return: LLM输出结果
    """
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(f"不支持的 prompt_type: {prompt_type}")
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=PROMPT_TEMPLATES[prompt_type]
    )
    chain = prompt | llm
    return chain.invoke({"text": text}).content

def extract_qa_pairs_from_llm_result(llm_result):
    """
    从LLM的输出结果中解析问答对
    """
    qa_pairs = []
    
    # 使用正则表达式匹配"问：...答：..."格式
    pattern = r'问：([^\n]*?)\n答：([^\n]*?)(?=\n问：|$)'
    matches = re.findall(pattern, llm_result, re.DOTALL)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        
        if question and answer:
            qa_pairs.append({
                '问': question,
                '答': answer
            })
    
    return qa_pairs

def split_text_to_qa_pairs(text):
    """
    智能提取"问...答..."结构的问答对，正确处理回答中包含"问"的情况。
    只提取第一个完整的问答对，忽略回答中的"问"字。
    """
    import re
    
    # 找到第一个"问"和第一个"答"的位置
    first_q_pos = text.find('问')
    first_a_pos = text.find('答')
    
    if first_q_pos == -1 or first_a_pos == -1 or first_a_pos <= first_q_pos:
        return []
    
    # 提取第一个问题（从第一个"问"到第一个"答"之间）
    question = text[first_q_pos + 1:first_a_pos].strip('：:，,。 　\n')
    
    # 提取第一个答案（从第一个"答"到文本结尾）
    answer = text[first_a_pos + 1:].strip('：:，,。 　\n')
    
    # 验证提取的内容
    if question and answer:
        return [{'问': question, '答': answer}]
    
    return []

def test_qa_extraction():
    """
    测试问答对提取功能，特别是处理回答中包含"问"的情况
    """
    test_text = """问您把事情的经过如实陈述一下答我是今天早上也就是二零二二年六月十四日六七点钟从叉叉市家中乘坐高铁前往武汉 然后在下午十四时许从武汉市乘坐高铁到达了杭州东站我这次来杭州就是为了前往我的前雇主阿雅家中索要薪资的大概在十七时许的样子我到了杭州市叉叉区钻石城二幢楼下因为我之前也找过阿雅要过工资的但是因为我文化学历不怎么样每次都说不过阿雅所以每次都没有要回我的全部工资所以我就联系了我的外甥阿伟一起来了钻石城我们碰头之后去了二幢二单元七零二室阿雅的家我在门口的时候房门是虚掩的我还敲了一下门阿雅问我是谁我回答他说小雅是我然后我就把房门打开了这个时候阿雅的丈夫看见我就问我说谁让我进房门的我就说是和阿雅打过招呼的 然后阿雅就让我进去了我进门直接就和阿雅还有她的丈夫表明了我来的目的就是希望他们能把之前欠我的工资结一下但是阿雅她还有她丈夫因为和家政中介存在纠纷暂时还不愿意给我解决阿雅和她丈夫也提供了几个方法就是他们联系中介过来我们三方坐下来好好谈一下这个钱的问题或者就是让我直接去找劳动监察部门解决这个事我当时坐了一天的车子到杭州身上也没有一分钱我看他们还是不愿意结账给我我就情绪特别激动了我特别执着的想要今天就拿到这笔钱因为我在杭州确实也耗不起时间我没有地方住也没有钱吃饭然后沟通没有结果之后阿雅和她丈夫就想让我先离开然后明天会联系中介一起解决的我就不愿意走因为我身上没有钱之后阿雅就报警了你们警方到达现场之后也在劝我 但是当时我已经特别激动了所以我都没有理会你们传唤我的时候我也没有接受传唤后面就被你们民警强制传唤至派出所接受调查了"""
    
    print("=== 测试文本 ===")
    print(test_text)
    print("\n=== 问答对提取结果 ===")
    
    result = split_text_to_qa_pairs(test_text)
    
    if result:
        print(f"成功提取到 {len(result)} 个问答对：")
        for i, qa in enumerate(result, 1):
            print(f"\n问答对 {i}:")
            print(f"问: {qa['问']}")
            print(f"答: {qa['答'][:1000]}..." if len(qa['答']) > 1000 else f"答: {qa['答']}")
    else:
        print("未提取到任何问答对")
    
    print("\n=== 调试信息 ===")
    # 显示文本中的"问"和"答"位置
    import re
    q_positions = [m.start() for m in re.finditer(r'问', test_text)]
    a_positions = [m.start() for m in re.finditer(r'答', test_text)]
    print(f"文本中'问'的位置: {q_positions}")
    print(f"文本中'答'的位置: {a_positions}")
    print(f"总共找到 {len(q_positions)} 个'问'，{len(a_positions)} 个'答'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen LLM 测试接口")
    parser.add_argument('--text', type=str, required=False, default="问:你是谁？答:我是AI助手。问:你能做什么？答:我能帮你处理文本。", help="要处理的文本")
    parser.add_argument('--type', type=str, required=False, default="qa", help="处理类型，如qa/summary/keywords")
    parser.add_argument('--test', action='store_true', help="运行测试函数")
    args = parser.parse_args()
    
    if args.test:
        test_qa_extraction()
    else:
        print(f"输入文本: {args.text}")
        print("\n=== Qwen LLM 处理结果 ===\n")
        try:
            llm_result = process_text_with_qwen(args.text, args.type)
            print(llm_result)
            
            if args.type == "qa":
                print("\n=== 解析后的问答对 ===\n")
                qa_pairs = extract_qa_pairs_from_llm_result(llm_result)
                for i, qa in enumerate(qa_pairs, 1):
                    print(f"问答对 {i}:")
                    print(f"问: {qa['问']}")
                    print(f"答: {qa['答']}")
                    print()
        except Exception as e:
            print(f"处理出错: {e}")
            print("\n=== 使用本地函数作为备用 ===\n")
            print(split_text_to_qa_pairs(args.text)) 