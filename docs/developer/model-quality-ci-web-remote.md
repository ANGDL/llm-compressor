# Model Quality Web 远程连接操作手册

## 目标

客户端只需要记住一个命令：

```bash
scripts/model-quality-web-remote open
```

它会自动处理以下变化：

- SSH control socket 不存在、失效或电脑重启；
- 远端 Web 进程退出；
- 本地 SSH tunnel 断开；
- 容器、服务器、端口、代码路径或运行目录发生变化；
- 容器没有 `.venv`，但存在其他 Python 环境。

拓扑保持为：

```text
local browser → 127.0.0.1:<local-port>
              → SSH master/tunnel
              → remote node 127.0.0.1:<remote-port>
              → configured Docker container Web process
```

Web 服务只监听容器的 `127.0.0.1`，不能从远端网络直接访问。

## 首次配置

```bash
scripts/model-quality-web-remote init
```

编辑 `.model-quality/web-remote.env`：

```bash
MQ_SSH_HOST=node16
MQ_EXISTING_CONTROL_SOCKET=/tmp/codex-node16.sock
MQ_CONTAINER=llm-quant-base
MQ_REPO=/data/quant/llm-compressor
MQ_LOCAL_PORT=18080
MQ_REMOTE_PORT=18080
MQ_RUNS_ROOT=/ssd2/model-quality/runs
MQ_CONFIG=ci/model_quality/config/models.yaml
MQ_GIT_SHA=<deployed-git-sha>
```

配置文件位于被 git 忽略的 `.model-quality/`，可以保存每台机器的真实容器名和路径。
若需要管理多套服务，为每套服务保存独立配置：

```bash
scripts/model-quality-web-remote --config ~/.config/mq/node16.env open
scripts/model-quality-web-remote --config ~/.config/mq/node17.env open
```

两套配置应使用不同的 `MQ_LOCAL_PORT` 和 control socket。默认 control socket 为
`/tmp/model-quality-<ssh-host>.sock`。如果已存在经过认证的 Codex/jumpserver master，
配置 `MQ_EXISTING_CONTROL_SOCKET`；管理器会优先复用它，并在它消失后尝试交互式建立
自己的 socket。

## 日常操作

```bash
# 一次性检查 SSH、Docker、容器、仓库、manifest 和 Python
scripts/model-quality-web-remote doctor

# 自动恢复全部链路并打开浏览器
scripts/model-quality-web-remote open

# 仅建立链路，不打开浏览器
scripts/model-quality-web-remote up

# 查看每一层的状态
scripts/model-quality-web-remote status

# 查看远端服务日志
scripts/model-quality-web-remote logs

# 只修复 SSH tunnel
scripts/model-quality-web-remote tunnel

# 只修复容器内服务
scripts/model-quality-web-remote service
```

## 发生变化时

### 容器被重建或更名

修改：

```bash
MQ_CONTAINER=<new-container>
MQ_PYTHON=<python-inside-new-container-or-empty>
```

然后执行：

```bash
scripts/model-quality-web-remote doctor
scripts/model-quality-web-remote restart
```

`doctor` 会明确报告容器不存在、仓库未挂载、manifest 缺失或 Python 找不到。

### SSH socket 消失

无需手工处理。再次执行：

```bash
scripts/model-quality-web-remote open
```

管理器会删除无效 socket 并用 `ControlPersist=8h` 建立新 master。若 jumpserver
要求登录确认，此时 SSH 会正常显示它的认证流程。

### 更换服务器

修改 `MQ_SSH_HOST`。默认 control socket 会跟随 host 改名，然后运行 `doctor` 和
`open`。如果新服务器上容器、repo、runs root 不同，同时修改对应字段。

### 更换服务代码或端口

代码更新后修改 `MQ_GIT_SHA`，必要时修改 `MQ_REMOTE_PORT`，然后执行：

```bash
scripts/model-quality-web-remote restart
```

若本地端口冲突，只改 `MQ_LOCAL_PORT`。浏览器地址也会自动随之变化。

### 客户端关闭但服务继续运行

```bash
scripts/model-quality-web-remote down
```

该命令默认只取消 tunnel。若必须同时停止远端服务：

```bash
MQ_STOP_SERVICE=1 scripts/model-quality-web-remote down
```

## 长期运行

客户端管理器适合开发和单人使用。如果 Web 服务必须在没有客户端 SSH 会话时保持
运行，由服务器管理员安装 `ci/model_quality/web/deploy/model-quality-web.service`：

1. 将环境配置安装为 `/etc/model-quality/web.env`；
2. 确保 systemd 中 Docker 已启动；
3. `systemctl enable --now model-quality-web`；
4. 客户端仍使用管理器的 `tunnel` 或 `open` 建立本地访问。

systemd 只管理服务进程；容器仍需由既有 Docker/编排平台维护。容器重建后，unit 的
`Restart=always` 会重试，容器名称变化则更新 `/etc/model-quality/web.env`。

## 故障定位

按顺序执行：

```bash
scripts/model-quality-web-remote doctor
scripts/model-quality-web-remote status
scripts/model-quality-web-remote logs
```

常见结果：

- `SSH master unavailable`：重新执行 `open` 并完成 SSH/jumpserver 认证；
- `container missing/stopped`：更新 `MQ_CONTAINER` 或启动容器；
- `repository missing`：修复容器 mount 或更新 `MQ_REPO`；
- `manifest missing`：更新 `MQ_CONFIG`，它相对于 `MQ_REPO`；
- `remote Web service failed its health check`：查看 `logs`；
- `tunnel exists but health is unreachable`：检查端口冲突，修改 `MQ_LOCAL_PORT`；
- 页面只显示只读功能：使用 Identity 配置开发身份，生产环境由可信反向代理注入身份。
