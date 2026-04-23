# -*- coding: utf-8 -*-
"""
Banana 图像生成工具 - 使用 Gemini API 生成图像
"""
import base64
import requests
from typing import Any, Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage


class GenerateImageTool(Tool):
    """图像生成工具 - 使用 Gemini API 生成图像"""

    # 比例到分辨率的映射 (基于 1024px 基准)
    ASPECT_RATIO_MAP = {
        "1:1": (1024, 1024),
        "3:4": (768, 1024),
        "4:3": (1024, 768),
        "9:16": (576, 1024),
        "16:9": (1024, 576),
    }

    # 质量倍率
    QUALITY_SCALE = {
        "standard": 1.0,
        "2k": 1.5,
        "4k": 2.0,
    }

    def _invoke(
        self,
        tool_parameters: dict[str, Any],
        user_id: str,
        conversation_id: str | None = None,
        app_id: str | None = None,
        message_id: str | None = None,
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        调用图像生成工具
        """
        # 获取参数
        prompt = tool_parameters.get("prompt", "")
        aspect_ratio = tool_parameters.get("aspect_ratio", "1:1")
        quality = tool_parameters.get("quality", "standard")
        model = tool_parameters.get("model", "gemini-2.0-flash-exp-image-generation")

        if not prompt:
            yield self.create_text_message("错误: 请提供图像提示词")
            return

        # 计算图像尺寸
        width, height = self._calculate_dimensions(aspect_ratio, quality)

        # 从 Provider 凭证中获取 API Key
        api_key = self.runtime.credentials.get("gemini_api_key", "")

        if not api_key:
            yield self.create_text_message("错误: 未配置 API Key")
            return

        try:
            # 调用 API 生成图像
            yield self.create_text_message(f"正在生成图像 ({aspect_ratio}, {quality})...")
            image_data = self._call_api(prompt, width, height, model, api_key, quality)

            # 返回图像
            yield self.create_blob_message(
                blob=image_data,
                meta={
                    "mime_type": "image/png",
                    "width": width,
                    "height": height,
                },
            )

        except Exception as e:
            yield self.create_text_message(f"图像生成失败: {str(e)}")

    def _calculate_dimensions(self, aspect_ratio: str, quality: str) -> tuple:
        """
        根据比例和质量计算图像尺寸

        Args:
            aspect_ratio: 宽高比例
            quality: 质量等级

        Returns:
            (宽度, 高度)
        """
        # 获取基础尺寸
        base_width, base_height = self.ASPECT_RATIO_MAP.get(aspect_ratio, (1024, 1024))

        # 应用质量倍率
        scale = self.QUALITY_SCALE。get(quality, 1.0)
        width = int(base_width * scale)
        height = int(base_height * scale)

        # 确保不超过 API 限制 (最大 2048)
        max_size = 2048
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            width = int(width * ratio)
            height = int(height * ratio)

        return (width, height)

    def _call_api(
        self,
        prompt: str,
        width: int,
        height: int,
        model: str,
        api_key: str,
        quality: str,
    ) -> bytes:
        """
        调用 Gemini API 生成图像
        """
        # 构建请求体
        request_body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Generate an image: {prompt}. Image size: {width}x{height}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["Text", "Image"],
            },
        }

        # API 端点
        url = f"https://api.chipcloud.cc/v1beta/models/{model}:generateContent"

        # 请求头
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        # 发送请求
        response = requests.post(url, headers=headers, json=request_body)

        if response.status_code != 200:
            raise Exception(f"API 请求失败: {response.status_code} - {response.text}")

        # 解析响应
        result = response.json()

        # 提取图像数据
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "inlineData" in part:
                        image_data = base64.b64decode(part["inlineData"]["data"])
                        return image_data

        raise Exception("未能从 API 响应中提取图像数据")
