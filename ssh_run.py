"""AutoDL 服务器 SSH 辅助工具: 执行命令 / 上传文件。

用法:
    python ssh_run.py cmd "nvidia-smi"
    python ssh_run.py upload <local> <remote>        # 单文件
    python ssh_run.py upload_dir <local> <remote>    # 目录
    python ssh_run.py download <remote> <local>
"""
import getpass
import json
import os
import stat
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


def run(c, cmd, timeout=600):
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def upload(c, local, remote):
    sftp = c.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    print(f"uploaded {local} -> {remote}")


def upload_dir(c, local, remote):
    sftp = c.open_sftp()
    for root, dirs, files in os.walk(local):
        rel = os.path.relpath(root, local)
        rdir = remote if rel == "." else f"{remote}/{rel.replace(os.sep, '/')}"
        try:
            sftp.stat(rdir)
        except FileNotFoundError:
            sftp.mkdir(rdir)
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            lp = os.path.join(root, f)
            rp = f"{rdir}/{f}"
            sftp.put(lp, rp)
            print(f"  {lp} -> {rp}")
    sftp.close()
    print(f"uploaded dir {local} -> {remote}")


def download(c, remote, local):
    sftp = c.open_sftp()
    sftp.get(remote, local)
    sftp.close()
    print(f"downloaded {remote} -> {local}")


if __name__ == "__main__":
    c = connect()
    try:
        cmd = sys.argv[1]
        if cmd == "cmd":
            cmdline = sys.argv[2]
            if len(sys.argv) > 3 and sys.argv[3].isdigit():
                timeout = int(sys.argv[3])
            else:
                timeout = 600
            _, out, err = run(c, cmdline, timeout=timeout)
            print("=== stdout ===")
            print(out)
            if err.strip():
                print("=== stderr ===")
                print(err)
        elif cmd == "upload":
            upload(c, sys.argv[2], sys.argv[3])
        elif cmd == "upload_dir":
            upload_dir(c, sys.argv[2], sys.argv[3])
        elif cmd == "download":
            download(c, sys.argv[2], sys.argv[3])
        else:
            print("unknown cmd")
    finally:
        c.close()