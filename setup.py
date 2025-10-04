from setuptools import find_packages, setup

setup(
    name="gui-agents",
    version="0.3.1",
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
        "anthropic",
        "fastapi",
        "uvicorn",
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
    entry_points={
        "console_scripts": [
            "agent_s=gui_agents.s3.cli_app:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
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
    python_requires=">=3.9, <=3.12",
)
