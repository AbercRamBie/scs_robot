from setuptools import find_packages, setup

package_name = 'semcomm_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='subash',
    maintainer_email='subashram773@gmail.com',
    description='Semantic Communication System - core models, training, and evaluation utilities',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_semcomm = semantic_comm_core.run_train:main',
        ],
    },
)
