from setuptools import setup, find_packages

version = "0.0.2"

extra_requirements = [
    "hidapi",
    "numpy>=1.17.4",
    "hid==1.0.4",
    "pyglm",
    "scipy",
    "ansitable",
    "progress",
    "typing_extensions",
    "ipykernel",
    "matplotlib"
]

install_requires = extra_requirements

setup(
    name='joycon-robotics',
    version=version,
    description='Joystick Controller for Robotics',
    author='boxjod, Huanxu Lin',
    author_email=', '.join([
        'boxjod@163.com',
        "linhxforeduct@outlook.com"
    ]),
    url='https://github.com/box2ai-robotics/joycon-robotics',
    packages=find_packages(),
    install_requires=install_requires,  # 使用合并后的依赖列表
    classifiers=[
        'Programming Language :: Python :: 3.7'
    ]
)

