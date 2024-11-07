from setuptools import setup, find_packages

setup(
    name='agent_s',
    version='0.1.0',
    description='A library for creating general purpose GUI agents using multimodal LLMs.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Saaket Agashe',
    author_email='saagashe@ucsc.edu',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas', 
        'openai',
        'torch',
        'torchvision',
        'transformers',
        'anthropic',
        'fastapi',
        'uvicorn',
        'paddleocr',
        'paddlepaddle',
        'together',
        'scikit-learn',
        'websockets',
        'tiktoken',
        'pyobjc; platform_system == "Darwin"',
        'pyautogui'
    ],
    entry_points={
        'console_scripts': [
            'agent_s=agent_s.cli_app:main',
        ],
    },
     classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache License',
        'Operating System :: MacOS',
        'Operating System :: Ubuntu',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='ai, llm, gui, agent, multimodal',
    project_urls={
        'Source': 'https://github.com/saaketagashe/Agent-S',
        'Bug Reports': 'https://github.com/saaketagashe/Agent-S/issues',
    },
    python_requires='>=3.9',
)
