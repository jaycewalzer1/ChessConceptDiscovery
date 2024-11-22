from setuptools import setup

setup(
    name="AlphaZeroChess",
    version="0.1",
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy",
        "open_spiel",
        "gym"
    ],
    packages=[
        'agents',
        'envs',
        'scripts'
    ],
    entry_points={
        'console_scripts': [
            'run_training=scripts.run_training:run_training',
            'eval_agent=scripts.eval_agent:eval_agent'
        ]
    }
)
