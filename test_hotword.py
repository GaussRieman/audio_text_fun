#!/usr/bin/env python3
"""
热词功能测试脚本
测试空间分隔的热词格式是否正确工作
"""

import os
import sys
from model import ParaformerModel, test_hotword_functionality

def main():
    """主测试函数"""
    print("=== 热词功能测试 ===")
    
    # 测试音频文件路径
    audio_path = "assets/vad_example.wav"
    
    if not os.path.exists(audio_path):
        print(f"❌ 测试音频文件不存在: {audio_path}")
        print("请确保 assets/vad_example.wav 文件存在")
        return
    
    # 测试用例
    test_cases = [
        {
            "name": "无热词测试",
            "hotword": None,
            "description": "不使用热词的基础转写"
        },
        {
            "name": "单个热词测试",
            "hotword": "是错",
            "description": "使用单个热词"
        },
        {
            "name": "多个热词测试",
            "hotword": "是错 试验",
            "description": "使用多个空格分隔的热词"
        },
        {
            "name": "多个热词测试2",
            "hotword": "语音 识别 测试",
            "description": "使用更多空格分隔的热词"
        }
    ]
    
    # 执行测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i}: {test_case['name']} ---")
        print(f"描述: {test_case['description']}")
        print(f"热词: {test_case['hotword'] if test_case['hotword'] else '无'}")
        
        try:
            result = test_hotword_functionality(audio_path, test_case['hotword'])
            
            if result['success']:
                print(f"✅ {test_case['name']} 成功")
                print(f"   转写结果: {result['transcribed_text'][:100]}...")
                print(f"   处理时间: {result['processing_time']:.2f}秒")
            else:
                print(f"❌ {test_case['name']} 失败")
                print(f"   错误信息: {result.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"❌ {test_case['name']} 异常: {str(e)}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main() 