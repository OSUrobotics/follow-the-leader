from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'follow_the_leader'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='main',
    maintainer_email='youa@oregonstate.edu',
    description='A package for a controller that scans tree branches by following up their main leader branch.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_processor = follow_the_leader.image_processor:main',
            'curve_fitting = follow_the_leader.curve_fitting:main',
            'controller = follow_the_leader.controller:main',
            'point_tracker = follow_the_leader.point_tracker:main',
            'gui = follow_the_leader.gui:main',
        ],
    },
)
