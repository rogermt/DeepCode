"""
Patch utilities for DeepCode compatibility fixes

This module provides configurable patches for third-party library compatibility issues.
"""

import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = "mcp_agent.config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the loaded configuration
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}

def apply_cerebras_patch() -> None:
    """
    Apply Cerebras compatibility patch if enabled in configuration.
    
    Checks the tool_calling.cerebras_compatibility setting and applies
    the monkey patch to fix tool response format issues with Cerebras.
    """
    config = load_config()
    
    if config.get("tool_calling", {}).get("cerebras_compatibility", False):
        # Import here to avoid circular imports
        from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
        from openai.types.chat import ChatCompletionToolMessageParam
        from mcp.types import TextContent, ImageContent, EmbeddedResource, TextResourceContents  
        import json
        
        # Store original method
        _original_execute_tool_call = OpenAIAugmentedLLM.execute_tool_call
        
        async def patched_execute_tool_call(self, tool_call):  
            """  
            Patched version that converts content arrays to plain strings  
            for OpenAI-compatible models like Cerebras.  
            """  
            tool_name = tool_call.function.name  
            tool_args_str = tool_call.function.arguments  
            tool_call_id = tool_call.id  
            tool_args = {}  
            
            try:  
                if tool_args_str:  
                    tool_args = json.loads(tool_args_str)  
            except json.JSONDecodeError as e:  
                return ChatCompletionToolMessageParam(  
                    role="tool",  
                    tool_call_id=tool_call_id,  
                    content=f"Invalid JSON provided in tool call arguments for '{tool_name}'. Failed to load JSON: {str(e)}",  
                )  
            
            from mcp.types import CallToolRequest, CallToolRequestParams  
            tool_call_request = CallToolRequest(  
                method="tools/call",  
                params=CallToolRequestParams(name=tool_name, arguments=tool_args),  
            )  
            
            result = await self.call_tool(  
                request=tool_call_request, tool_call_id=tool_call_id  
            )  
            
            # Convert content blocks to plain string  
            text_parts = []  
            for content in result.content:  
                if isinstance(content, TextContent):  
                    text_parts.append(content.text)  
                elif isinstance(content, ImageContent):  
                    text_parts.append(f"[Image: {content.mimeType}]")  
                elif isinstance(content, EmbeddedResource):  
                    if isinstance(content.resource, TextResourceContents):  
                        text_parts.append(content.resource.text)  
                    else:  
                        text_parts.append(f"[Resource: {content.resource.mimeType}]")  
            
            # Join all text parts with newlines  
            content_str = "\n\n".join(text_parts) if text_parts else ""  
            
            return ChatCompletionToolMessageParam(  
                role="tool",  
                tool_call_id=tool_call_id,  
                content=content_str,  # Plain string instead of array  
            ) 
            
        # Apply the monkey-patch
        OpenAIAugmentedLLM.execute_tool_call = patched_execute_tool_call
        print("✅ Applied Cerebras compatibility patch")
    else:
        print("ℹ️ Cerebras compatibility patch disabled")