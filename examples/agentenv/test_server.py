"""Offline checks; does not import SDKs or connect to a sandbox."""
import unittest
from types import SimpleNamespace
from server import register_tools


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tools = {}
        self.calls = []

        def tool():
            def register(fn):
                self.tools[fn.__name__] = fn
                return fn
            return register

        def run(command, timeout):
            self.calls.append((command, timeout))
            return SimpleNamespace(stdout="hello\n", stderr="", exit_code=0)

        register_tools(SimpleNamespace(tool=tool),
                       SimpleNamespace(commands=SimpleNamespace(run=run)))

    def test_multiline_command_and_timeout_reach_sandbox_unchanged(self):
        command = "printf '你好'\n  pwd"
        result = self.tools["run_command"](command, 10)
        self.assertEqual(self.calls, [(command, 10)])
        self.assertEqual(result, {"stdout": "hello\n", "stderr": "", "exit_code": 0})

    def test_bad_arguments_do_not_execute(self):
        for command, timeout in [("", 10), ("x", 0), ("x", 26), ("x" * 65537, 10)]:
            with self.assertRaises(ValueError):
                self.tools["run_command"](command, timeout)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
