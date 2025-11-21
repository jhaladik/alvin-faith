"""
HuggingFace Spaces Entry Point
Context-Aware Foundation Agent Demo
"""

from gradio_demo import create_demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
