import subprocess as sp
import shlex
import os
import pathlib


def main():

    file_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'utils', 'blender_server.py')
    cmd = 'blender --background --python {}'.format(file_path)
    sp.call(cmd, shell=True)


if __name__ == '__main__':
    main()
