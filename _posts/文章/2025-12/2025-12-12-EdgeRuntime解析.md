---
title: EdgeRuntime解析
date: 2025-12-12
categories: [文章, '202512']
tags: [supabase]
---

## 项目概述

这是一个基于 Rust 和 Deno 的项目，主要用于运行 JavaScript/TypeScript 代码在边缘计算环境中。

## 线程情况

1. v8初始化 初始建立cpu/4个线程(这里如果在容器 cpu取的是宿主机核数) 
    ● 即时编译 (JIT): 在后台将 JavaScript 代码编译成高效的本地机器码。
    ● 垃圾回收 (Garbage Collection): 执行并行的垃圾回收任务，减少主线程因 GC 而暂停的时间。
    ● 代码优化与反优化: 根据代码的运行情况，在后台进行激进的优化或在必要时回退。
    ● 后台解析: V8 可以在后台线程中解析 JavaScript 脚本，从而加快启动速度。
3. tokio::runtime::Builder::new_current_thread().thread_name("sb-main") 
   创建一个主线程 在主线程启动服务
4. tokio::runtime::Builder::new_current_thread().thread_name("sb-inspector") 
   单线程 监听调试连接，让你能够通过浏览器或 IDE 挂载断点、查看变量和分析内存
5. tokio::runtime::Builder::new_multi_thread().thread_name("sb-supervisor") 
   多线程(容器cpu核数一致) 监控请求的执行时间，处理超时，并在任务完成后回收资源
6. tokio::runtime::Builder::new_multi_thread().thread_name("io") 多线程(容器cpu核数一致) 
7. PRIMARY_WORKER_RT  tokio_util::task::LocalPoolHandle::new 
   启动配置数目的线程, 不用上面方法是因为要固定任务在当前线程
   这个（线程）池是专门给 Main Worker（主工作线程）和 Event Worker（事件工作线程）使用的。之所以要将它们与 User Worker（用户工作线程）池分开，是因为如果用户线程达到饱和状态，可能会导致主线程和事件线程发生‘饥饿’现象
8. USER_WORKER_RT  tokio_util::task::LocalPoolHandle::new
    启动配置数目的线程, 不用上面方法是因为要固定任务在当前线程
    每创建一个worker 就是一个v8 isolate (轻量)  然后取一个USER_WORKER_RUNTIME的空闲线程 固定v8 isolate到线程运行
9.  tokio-runtime-worker tokio默认运行时线程

## 启动流程

### 1. 进程启动
当运行 `edge-runtime start` 命令时：

1. **启动**：在 `cli/src/main.rs` 中启动一个Web Server(Tokio) 
2. **服务器构建**：创建 `Server` 实例，绑定到指定 IP 和端口
3. **主 Worker 创建**：启动主 Worker（Main Worker）
4. **可选的 Event Worker**：如果配置了事件处理，会启动 Event Worker

### 2. V8 实例数量
- **主 Worker**：1 个 V8 实例，运行主服务代码（如 `examples/main/index.ts`）
- **Event Worker**：0-1 个 V8 实例（可选）
- **User Workers**：根据请求动态创建，每个Worker一个 V8 实例


## 请求处理流程

### 1. 请求到达
```typescript
// 在 examples/main/index.ts 中
Deno.serve(async (req: Request) => {
  // 解析请求，确定服务名称
  // 创建或复用 User Worker
  // 转发请求到 User Worker
});
```

### 2. 服务路由
- 从域名和路径解析出 `service_name`
- 例如：`https://ref.backend.onspace.ai/functions/v1/functionname` → `ref-functionname`
- 查找对应的服务目录：`/home/vagrant/edge-runtime/examples/ref-functionname`

### 3. Worker 管理
- **Worker Pool**：管理所有 User Worker 的生命周期
- **Worker 策略**：
  - `per_worker`：每个服务路径保持一个Worker实例 (forceCreate配置false/ workerTimeoutMs配置短时间 / max-parallelism 可以对服务路径维度允许限制最大worker数  但是极端情况会不更新函数 - 因为进程是执行完空闲workerTimeoutMs才会被回收 因此空闲时一直有请求 进程就一直不销毁)
  - `per_request`：每个请求创建新的 Worker, 请求完成后放回连接池 (无法限制最大worker数)
  - `oneshot`：每个请求创建新 Worker，请求完成后销毁

### 4. User Worker 创建
```rust
// 在 crates/base/src/worker/pool.rs 中
pub fn create_user_worker(
  &mut self,
  mut worker_options: WorkerContextInitOpts,
  tx: Sender<Result<CreateUserWorkerResult, Error>>,
  termination_token: Option<TerminationToken>,
) {
  // 1. 检查是否有可用的 Worker
  // 2. 等待 Worker 空闲或创建新的
  // 3. 启动新的 V8 实例
  // 4. 加载用户代码
}
```

### 5. 请求转发
- 通过 **Duplex Stream** 将 HTTP 请求转发到 User Worker
- User Worker 处理请求并返回响应
- 支持 WebSocket 升级等高级功能

## 内存和性能管理

- **内存限制**：每个 Worker 默认 150MB 内存限制
- **CPU 时间限制**：软限制 10s，硬限制 20s
- **Worker 超时**：30s 超时
- **并发控制**：通过信号量控制最大并行 Worker 数量

## 总结

**多线程 + 多 V8 实例**的架构：
1. **启动时**：1-2 个 V8 实例（主 Worker + 可选的 Event Worker）
2. **运行时**：根据请求动态创建 User Worker，每个 Worker 一个 V8 实例
3. **请求处理**：主 Worker 接收请求 → 路由到对应服务 → 创建/复用 User Worker → 处理请求 → 返回响应

这种设计既保证了隔离性（每个用户代码在独立的 V8 实例中运行），又提供了灵活性（支持不同的 Worker 策略），适合边缘计算场景下的多租户服务部署。

