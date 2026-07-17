---
name: release
description: "Source release branching strategy for the llm-compressor fork. USE FOR: freezing main as a stable source release, tagging a version, promoting a prepared development branch to main, resolving upstream tag conflicts, renaming branches for release. DO NOT USE FOR: publishing PyPI packages, general git operations, pull request workflows, daily development."
---

# Release Process

本流程只管理 fork 的源码引用（branch 和 tag），不构建或发布 PyPI 包。若本次发布还包含 Python 包，必须在打 tag 前另行完成 release build、产物验证和发布审批。

## Branching Strategy

```text
upstream/main            上游原始分支（只读，不修改）

origin:
  main                   开发主分支
  stable/k-v{version}    冻结的稳定分支（历史存档）
  k-v{version}           tag，指向对应的稳定版本
```

## When to Release

- `main` 已达到稳定状态，需要冻结并发布正式源码版本
- 后续开发已在一个独立分支中进行并推送到 `origin`
- 该开发分支将被提升为新的 `main`

本文使用 `transformers-v5` 作为待提升开发分支的示例。实际操作时必须将 `NEXT_MAIN_BRANCH` 替换为本次待提升为新 `main` 的分支名称。

## Release Checklist

### 1. 完成发布准备

先在日常开发 checkout 中完成内容整理，不要一边整理文件一边创建 release tag。

```bash
# 示例；替换为本次待提升为新 main 的开发分支。
NEXT_MAIN_BRANCH=transformers-v5
```

#### 1.1 审查工作区

```bash
# 已跟踪修改必须明确属于旧稳定版、新开发主线，或两者。
git status --short --branch
git diff --stat
git diff --cached --stat

# 未跟踪文件必须逐项审查，不能只看顶层目录。
git ls-files --others --exclude-standard
```

若存在未跟踪文件，必须逐项分析，不能自动执行 `git add`、删除文件或修改 `.gitignore`：

- 判断文件用途以及是否属于本次稳定版本
- 检查是否为临时输出、日志、缓存、生成物或本地调试脚本
- 检查是否包含凭据、token、用户数据、模型数据或其他不应提交的内容
- 检查是否包含本机绝对路径或其他不可移植配置
- 检查大文件的来源、许可和敏感内容，以及是否应通过 Git LFS、外部存储或生成脚本管理
- 将文件分为“建议提交”“建议保持未跟踪”“需要人工进一步确认”三类，并说明理由
- 由发布负责人逐项确认哪些文件需要提交

决定纳入发布的文件应按功能形成独立提交。决定保持未跟踪的文件可以留在日常开发 checkout，但不得带入最终发版 checkout。

#### 1.2 将发布基础设施和共享修复同步到两条分支

发布前产生的 release skill、版本解析、测试以及两条线都需要的修复，必须明确合入旧 `main` 和待提升开发分支。不要假设提交存在于当前 checkout 就会自动进入两个最终 commit。

```bash
# 分别在两个分支上检查关键发布文件及其最近提交。
git log -n 5 --oneline main -- \
  .github/skills/release/SKILL.md src/llmcompressor/_version.py
git log -n 5 --oneline "origin/${NEXT_MAIN_BRANCH}" -- \
  .github/skills/release/SKILL.md src/llmcompressor/_version.py
```

如果修改需要同时存在于两条线，使用正常 merge 或有审查记录的 cherry-pick，并分别运行对应测试；不要在 release 操作期间临时复制工作区文件。

#### 1.3 使用干净 clone 执行最终发版

整理和提交完成后，最终的引用检查、打 tag、推送和本地分支重命名必须在专门的新 clone 中执行。这样可以避免日常 checkout 中的未跟踪数据、调试脚本和本地生成物干扰 `git switch` 或发布构建。

```bash
ORIGIN_URL=$(git remote get-url origin)
UPSTREAM_URL=$(git remote get-url upstream)
RELEASE_WORKDIR=../llm-compressor-release

# RELEASE_WORKDIR 必须不存在；不要复用旧目录。
test ! -e "$RELEASE_WORKDIR"
git clone "$ORIGIN_URL" "$RELEASE_WORKDIR"
cd "$RELEASE_WORKDIR"
git remote add upstream "$UPSTREAM_URL"
```

以下步骤均在这个专用 clone 中执行。

### 2. 设置本次发布参数

以下命令假设在同一个专用 shell 会话中执行。任一步失败都会终止该会话。默认版本将在步骤 3 中根据 `main` 最近可达的上游版本 tag 计算；只有确需覆盖默认值时才填写 `RELEASE_VERSION_OVERRIDE`：

```bash
set -euo pipefail

RELEASE_VERSION_OVERRIDE=
NEXT_MAIN_BRANCH=transformers-v5
# 第一次 fork 发布时留空；后续填写上一个 k-v* tag。
PREVIOUS_RELEASE_TAG=

test "$NEXT_MAIN_BRANCH" != main || {
  echo "NEXT_MAIN_BRANCH 必须是待提升的开发分支，不能是 main"
  exit 1
}
```

默认值取 `main` Git 历史中最近可达的非 `k-v*` 数字版本 tag，而不是全仓库版本号最高的 tag。例如，若 `main` 最近可达 tag 是 `0.12.0`，默认发布 tag 就是 `k-v0.12.0`。`v0.12.0` 会先归一化为 `0.12.0`。人工覆盖只用于有明确版本策略的特殊情况，并必须记录原因。

`RELEASE_VERSION` 是 fork 源码版本。若还要发布 Python 包，必须确认归一化后的版本未在目标包索引中使用，并按独立的包发布流程处理；不能因为 `k-v*` tag 尚不存在就认为包版本也可用。

### 3. 刷新引用并确认现状

```bash
git fetch origin --prune --tags
git fetch upstream --tags \
  "+refs/heads/main:refs/remotes/upstream/main"
git switch main

# 发布必须基于已推送的 main，避免 tag 指向仅存在于本地的提交。
test "$(git rev-parse main)" = "$(git rev-parse origin/main)" || {
  echo "本地 main 与 origin/main 不一致"
  exit 1
}

# 只考虑 main 可达的上游风格版本 tag；排除之前的 fork release tag。
BASE_VERSION_TAG=$(git describe --tags --abbrev=0 \
  --exclude 'k-v*' \
  --match '[0-9]*' \
  --match 'v[0-9]*' \
  main) || {
  echo "main 历史中找不到可用的版本 tag"
  exit 1
}
DEFAULT_RELEASE_VERSION=${BASE_VERSION_TAG#v}
RELEASE_VERSION=${RELEASE_VERSION_OVERRIDE:-$DEFAULT_RELEASE_VERSION}

# 当前上游历史包含三段和四段数字版本，两种都允许继承。
if [[ ! "$RELEASE_VERSION" =~ ^[0-9]+(\.[0-9]+){2,3}$ ]]; then
  echo "推导出的 RELEASE_VERSION 不是支持的数字版本: ${RELEASE_VERSION}"
  exit 1
fi

RELEASE_TAG="k-v${RELEASE_VERSION}"
STABLE_BRANCH="stable/${RELEASE_TAG}"
printf 'base version tag: %s\nrelease tag: %s\n' \
  "$BASE_VERSION_TAG" "$RELEASE_TAG"

# 确认没有已跟踪文件的修改。
git diff --quiet && git diff --cached --quiet || {
  echo "请先提交或还原已跟踪文件的修改"
  exit 1
}

# 专用发版 clone 不应包含任何未跟踪文件。
UNTRACKED_FILES=$(git ls-files --others --exclude-standard)
if test -n "$UNTRACKED_FILES"; then
  printf '专用发版 clone 中出现未跟踪文件，流程终止：\n%s\n' \
    "$UNTRACKED_FILES"
  exit 1
fi

# tag、远程稳定分支和本地稳定分支必须尚不存在。
git show-ref --verify --quiet "refs/tags/${RELEASE_TAG}" && {
  echo "tag ${RELEASE_TAG} 已存在"
  exit 1
}
if git ls-remote --exit-code --heads origin "$STABLE_BRANCH" >/dev/null 2>&1; then
  echo "远程分支 ${STABLE_BRANCH} 已存在"
  exit 1
else
  status=$?
  test "$status" -eq 2 || {
    echo "无法查询远程分支 ${STABLE_BRANCH}"
    exit "$status"
  }
fi
git show-ref --verify --quiet "refs/heads/${STABLE_BRANCH}" && {
  echo "本地分支 ${STABLE_BRANCH} 已存在"
  exit 1
}

# 待提升分支必须已推送。若本地也存在，必须与远程同步。
git show-ref --verify --quiet "refs/remotes/origin/${NEXT_MAIN_BRANCH}" || {
  echo "origin/${NEXT_MAIN_BRANCH} 不存在"
  exit 1
}
if git show-ref --verify --quiet "refs/heads/${NEXT_MAIN_BRANCH}"; then
  test "$(git rev-parse "$NEXT_MAIN_BRANCH")" = \
    "$(git rev-parse "origin/$NEXT_MAIN_BRANCH")" || {
    echo "本地 ${NEXT_MAIN_BRANCH} 与 origin/${NEXT_MAIN_BRANCH} 不一致"
    exit 1
  }
fi

RELEASE_COMMIT=$(git rev-parse main)
NEXT_MAIN_COMMIT=$(git rev-parse "origin/${NEXT_MAIN_BRANCH}")
printf 'release commit: %s\nnext main commit: %s\n' \
  "$RELEASE_COMMIT" "$NEXT_MAIN_COMMIT"
```

### 4. 审查发布内容和分支分叉

```bash
# 后续版本优先相对上一个 fork release 审查；第一次发布才与 upstream 建立基线。
if test -n "$PREVIOUS_RELEASE_TAG"; then
  git show-ref --verify --quiet "refs/tags/${PREVIOUS_RELEASE_TAG}" || {
    echo "找不到上一个 release tag: ${PREVIOUS_RELEASE_TAG}"
    exit 1
  }
  git log --oneline "${PREVIOUS_RELEASE_TAG}..main"
  git diff --stat "${PREVIOUS_RELEASE_TAG}..main"
else
  git log --oneline upstream/main..main
  git diff --stat upstream/main..main
fi

# 两侧有独有提交并不一定是错误，但必须确认这是预期的版本边界。
git rev-list --left-right --count "main...origin/${NEXT_MAIN_BRANCH}"
git log --oneline --left-right "main...origin/${NEXT_MAIN_BRANCH}"

# 区分真正缺失的补丁与 SHA 不同但 patch 等价的提交。
git cherry "origin/${NEXT_MAIN_BRANCH}" main

# 对从共同祖先开始的两组提交做序列级比较。输出可能很长。
MERGE_BASE=$(git merge-base main "origin/${NEXT_MAIN_BRANCH}")
git range-diff "${MERGE_BASE}..main" \
  "${MERGE_BASE}..origin/${NEXT_MAIN_BRANCH}"
```

提交标题相同不表示实现等价；`git cherry` 显示 `+` 的旧 `main` 提交必须逐个决定是移植、重写还是只保留在稳定版。不要为消除分叉而直接 merge 整个旧 `main`，除非完整 diff 和测试证明这样做符合新主线设计。审查结论应记录在 PR、issue 或发布工单中，不要在专用发版 clone 内生成临时报告文件。

继续前必须人工确认：

- `RELEASE_COMMIT` 是要冻结的准确提交
- `NEXT_MAIN_COMMIT` 是新 `main` 的准确提交
- 所有未跟踪文件均已分析，并由发布负责人确认是否提交
- 两个最终 commit 都包含其需要的 release 基础设施和共享修复
- 旧 `main` 的独有提交只保留在稳定版本中是预期行为
- `git cherry` 中每个 `+` 提交都有明确处理结论
- 必需测试和 CI 已分别在精确的 `RELEASE_COMMIT` 和 `NEXT_MAIN_COMMIT` 上通过
- 需要进入稳定版本的 README、Release Notes 或链接修改已经提交到 `main`
- 已检查以旧 `main` 为 base 的开放 PR，并确定删除旧 `main` 后的处理方式
- 已检查 GitHub branch protection、ruleset、自动化和外部集成对删除或重命名 `main` 的限制

本仓库 CI 当前监听 `main` 和 `release/*`，不监听 `stable/k-v*`。因此不能等稳定分支创建后再补做验证；如开发分支未自动触发完整 CI，应先通过 PR 或临时 `release/*` 验证分支让 `NEXT_MAIN_COMMIT` 跑完相同检查。记录工作流链接和已验证 SHA，之后不能在未重跑 CI 的情况下移动候选分支。

### 5. 原子创建远程 Tag 和稳定分支

Tag 命名规则为 `k-v{version}`，其中 `{version}` 默认继承步骤 3 推导出的上游数字版本。`k-` 前缀避免与 upstream tag 冲突；项目版本代码会将 `k-v0.12.0` 归一化为 PEP 440 版本 `0.12.0`。

```bash
git tag -a "$RELEASE_TAG" "$RELEASE_COMMIT" \
  -m "Release ${RELEASE_TAG}"

# 两个远程引用同时成功或同时失败，避免只推送成功其中一个。
git push --atomic origin \
  "refs/tags/${RELEASE_TAG}" \
  "main:refs/heads/${STABLE_BRANCH}"
```

若原子 push 失败，两个远程引用都不会创建，但本地 annotated tag 仍然存在。排除失败原因并确认远程引用不存在后，可执行 `git tag -d "$RELEASE_TAG"`，然后从步骤 2 重新开始。

### 6. 在 GitHub 上切换开发主分支

在 GitHub 仓库 Settings -> Branches 中按顺序操作：

1. 再次确认 `$NEXT_MAIN_BRANCH` 仍指向已验证的 `$NEXT_MAIN_COMMIT`
2. 将默认分支暂时切换到 `$NEXT_MAIN_BRANCH`
3. 删除旧的 `main`；旧稳定提交已由 `$STABLE_BRANCH` 和 `$RELEASE_TAG` 保存
4. 将 `$NEXT_MAIN_BRANCH` 重命名为 `main`
5. 确认默认分支随重命名更新为新的 `main`
6. 重新检查依赖分支名或默认分支的 branch protection、ruleset、Actions、部署和外部集成
7. 保护 `$STABLE_BRANCH`：禁止直接 push、force push 和删除

不要将旧 `main` 重命名为 `$STABLE_BRANCH`：步骤 5 已创建该稳定分支，重名会导致 GitHub 拒绝操作。

### 7. 验证远程引用

annotated tag 本身指向 tag 对象；必须查询带 `^{}` 的 peeled 引用才能得到发布 commit。

```bash
REMOTE_MAIN_COMMIT=$(git ls-remote --heads origin \
  "refs/heads/main" | awk '{print $1}')
REMOTE_STABLE_COMMIT=$(git ls-remote --heads origin \
  "refs/heads/${STABLE_BRANCH}" | awk '{print $1}')
REMOTE_TAG_COMMIT=$(git ls-remote --tags origin \
  "refs/tags/${RELEASE_TAG}^{}" | awk '{print $1}')

test "$REMOTE_MAIN_COMMIT" = "$NEXT_MAIN_COMMIT"
test "$REMOTE_STABLE_COMMIT" = "$RELEASE_COMMIT"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
```

三个 `test` 都成功后，才能进行本地同步。

### 8. 本地同步并切换到新 main

```bash
git fetch origin --prune

# 当前本地 main 仍是已发布的旧 main。
git branch -m main "$STABLE_BRANCH"
git branch --set-upstream-to="origin/${STABLE_BRANCH}" "$STABLE_BRANCH"

if git show-ref --verify --quiet "refs/heads/${NEXT_MAIN_BRANCH}"; then
  git branch -m "$NEXT_MAIN_BRANCH" main
  git branch --set-upstream-to=origin/main main
  git switch main
else
  git switch -c main --track origin/main
fi

test "$(git branch --show-current)" = main
test "$(git rev-parse main)" = "$NEXT_MAIN_COMMIT"
```

最后两个检查可防止后续修改误落到冻结的稳定分支。

### 9. 通知用户

在 GitHub Release Notes 或 README 中说明：

- 新的 `main` 是开发主分支
- 稳定源码通过 tag `$RELEASE_TAG` 获取；通知用户时用实际版本替换占位符：`git clone -b k-v{version} <url>`
- 旧 `main` 保存在 `$STABLE_BRANCH`
- 已有 clone 的用户需执行：

```bash
git fetch origin --prune
git branch -m main stable/k-v{version}
git switch -c main --track origin/main
```

## 后续日常开发

```bash
git switch main
git pull --ff-only
git switch -c feature/xxx
# 开发完成后
git switch main
git merge --no-ff feature/xxx
git push origin main
```

## 注意事项

- 不要向 `stable/k-v{version}` 提交代码，该分支仅作为历史存档
- `k-v{version}` 是不可变的正式发布标识
- 不要将 tag 命名为裸的 `v*`，这会与 upstream 的 tag 冲突
- 不要用 `git reset --hard` 或 `--force` 覆盖稳定分支
- 只删除旧的 `main`，不要删除稳定分支
- `upstream/main` 不参与写操作，仅用于审查 fork 相对上游的发布内容
