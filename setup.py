from setuptools import find_packages, setup

setup(
    name="gui-agents",
    version="0.3.2",
    description="A library for creating general purpose GUI agents using multimodal LLMs.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Simular AI",
    author_email="eric@simular.ai",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "backoff",
        "pandas",
        "openai",
        "openai-codex>=0.160,<0.161",
        "anthropic",
        "fastapi",
        "uvicorn",
        "mcp>=2.2,<3",
        "httpx>=0.27,<1",
        "pydantic>=2.7,<3",
        "mss>=9,<11",
        "pyperclip>=1.8,<2",
        "paddleocr",
        "paddlepaddle",
        "together",
        "scikit-learn",
        "websockets",
        "tiktoken",
        "selenium",
        'pyobjc; platform_system == "Darwin"',
        "pyautogui",
        "toml",
        "pytesseract",
        "google-genai",
        'pywinauto; platform_system == "Windows"',  # Only for Windows
        'pywin32; platform_system == "Windows"',  # Only for Windows
    ],
    extras_require={"dev": ["black"]},  # Code formatter for linting
    package_data={
        "gui_agents.s3": ["ui/index.html", "ui/static/*.js", "ui/static/*.css"]
    },
    entry_points={
        "console_scripts": [
            "agent_s=gui_agents.s3.cli_app:main",
            "agent_s_ui=gui_agents.s3.ui_server:main",
            "agent_s_mcp=gui_agents.s3.mcp_server:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai, llm, gui, agent, multimodal",
    project_urls={
        "Source": "https://github.com/simular-ai/Agent-S",
        "Bug Reports": "https://github.com/simular-ai/Agent-S/issues",
    },
    python_requires=">=3.10,<3.13",
)
