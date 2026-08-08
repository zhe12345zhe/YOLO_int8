"""UTF-8 日志运行器: 以 UTF-8 运行子进程, 输出逐行写入 UTF-8 日志文件并实时回显。

用法: python tee_run.py --log out.log -- python train.py --epochs 50
      (-- 之后的所有内容原样作为要运行的命令)
输出文件为 UTF-8 with BOM (VS Code / Notepad 均直接可读), 全程不经过 PowerShell 转码。
"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 回显不因 GBK 崩溃, 文件才是唯一真源


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("用法: python tee_run.py --log <文件> -- <命令...>")
    dl, k = argv.index("--"), argv.index("--log")
    if k + 1 >= dl:
        sys.exit("错误: 缺少 --log 的文件名")
    log_path, cmd = argv[k + 1], argv[dl + 1:]
    if not cmd:
        sys.exit("错误: 未提供要运行的命令")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"   # 子进程 stdout/stderr 一律 UTF-8 字节
    env["PYTHONUTF8"] = "1"

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env, text=True, encoding="utf-8", errors="replace",
                         bufsize=1)
    with open(log_path, "w", encoding="utf-8-sig", buffering=1) as f:
        for line in p.stdout:
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
    p.wait()
    print(f"\n[tee_run] 退出码 {p.returncode}, 日志: {log_path}")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()