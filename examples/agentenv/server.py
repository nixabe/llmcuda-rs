"""Optional AgentENV-to-MCP bridge. See docs/MCP.md for setup.

Each stdio process creates its own sandbox. Credentials and the template are
administrator environment variables, never model-supplied tool arguments.
"""
import os


def register_tools(server, sandbox):
    @server.tool()
    def run_command(command: str, timeout_seconds: int = 20) -> dict:
        """Run a shell command inside this session's AgentENV sandbox."""
        if not command or len(command.encode("utf-8")) > 65536:
            raise ValueError("command must contain 1 to 65536 UTF-8 bytes")
        if not 1 <= timeout_seconds <= 25:
            raise ValueError("timeout_seconds must be between 1 and 25")
        result = sandbox.commands.run(command, timeout=timeout_seconds)
        output = {"stdout": result.stdout, "stderr": result.stderr,
                  "exit_code": result.exit_code}
        if len((result.stdout + result.stderr).encode("utf-8")) > 262144:
            raise ValueError("command output exceeds 256 KiB; command was executed")
        return output


def main():
    from e2b import Sandbox
    from mcp.server import MCPServer

    # Require explicit AgentENV configuration; never fall back to a cloud URL.
    for key in ("E2B_API_URL", "E2B_SANDBOX_URL", "E2B_API_KEY", "AENV_TEMPLATE"):
        if not os.environ.get(key):
            raise RuntimeError(f"missing {key}")
    sandbox = Sandbox.create(os.environ["AENV_TEMPLATE"], timeout=600)
    try:
        server = MCPServer("AgentENV session")
        register_tools(server, sandbox)
        server.run(transport="stdio")
    finally:
        # A forced process kill cannot run this block. The remote 600-second
        # lifetime remains the cleanup bound in that case.
        sandbox.kill()


if __name__ == "__main__":
    main()
