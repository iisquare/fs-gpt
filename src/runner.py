import os
import sys
import unittest

from fs_gpt import cli


class Runner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cwd = os.getcwd()
        print(f'cwd: {cwd}')
        cwd = os.path.dirname(cwd)
        print(f'workdir: {cwd}')
        os.chdir(cwd)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_train_pt_full(self):
        sys.argv = ["fs-gpt", "run", "examples/train_pt_full.yaml"]
        cli.main()

    def test_train_sft_lora(self):
        sys.argv = ["fs-gpt", "run", "examples/train_sft_lora.yaml"]
        cli.main()

    def test_vocab_append(self):
        sys.argv = ["fs-gpt", "run", "examples/vocab_append.yaml"]
        cli.main()

if __name__ == '__main__':
    unittest.main()
