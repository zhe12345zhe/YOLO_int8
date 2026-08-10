#!/usr/bin/env python
"""安全地通过 SSH 执行远端命令 (解决多层引号问题)。

用法: python ssh_run_safe.py <base64-bash-script>
"""
import base64
import json
import os
import sys
from pathlib import Path

import paramiko

_CRED = json.loads((Path(__file__).parent / "yolo-int8.json").read_text(encoding="utf-8"))
HOST = _CRED.get("host", os.environ.get("YOLO_SSH_HOST"))
PORT = int(_CRED.get("port", os.environ.get("YOLO_SSH_PORT", 22)))
USER = _CRED.get("user", "root")
PASSWORD = _CRED.get("password", os.environ.get("YOLO_SSH_PASSWORD", ""))


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    except Exception as e:
        print(f"连接失败: {e}\n"
              f"提示: 云端 GPU (AutoDL) 开机后 SSH 端口可能变化, "
              f"请按控制台「SSH 登录」信息更新 {Path(__file__).parent / 'yolo-int8.json'} 的 host/port 后重试")
        raise
    return c


def main():
    b64 = sys.argv[1]
    script = base64.b64decode(b64).decode("utf-8")
    # 由 /bin/bash 执行, 避免 shell 嵌套引号问题
    cmd = f"echo {b64} | base64 -d > /tmp/remote_cmd.sh && bash /tmp/remote_cmd.sh"
    c = connect()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=86400)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        if out:
            sys.stdout.buffer.write(out.encode("utf-8", "replace"))
            sys.stdout.flush()
        if err:
            print("=== stderr ===")
            sys.stdout.buffer.write(err.encode("utf-8", "replace"))
            sys.stdout.flush()
        print(f"exit={rc}")
        sys.exit(rc)
    finally:
        c.close()


if __name__ == "__main__":
    main()