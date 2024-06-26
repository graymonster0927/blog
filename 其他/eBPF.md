# eBPF


## BCC性能工具
![bpf_performance_tools_book.png](./image/bpf_performance_tools_book.png)

## bpftrace性能工具
![bpftrace_tools_early2019.png](./image/bpftrace_tools_early2019.png)

### 工具作用

* argdist:     Display function parameter values as a histogram or frequency count.

* bashreadline:     Print entered bash commands system wide.

* biolatency:     Summarize block device I/O latency as a histogram.

* biotop:     Top for disks:     Summarize block device I/O by process.

* biosnoop:     Trace block device I/O with PID and latency.

* bitesize:     Show per process I/O size histogram.

* bpflist:     Display processes with active BPF programs and maps.

* btrfsdist:     Summarize btrfs operation latency distribution as a histogram.

* btrfsslower:     Trace slow btrfs operations.

* capable:     Trace security capability checks.

* cachestat:     Trace page cache hit/miss ratio.

* cachetop:     Trace page cache hit/miss ratio by processes.

* compactsnoop:     Trace compact zone events with PID and latency.

* cpudist:     Summarize on- and off-CPU time per task as a histogram. Examples

* cpuunclaimed:     Sample CPU run queues and calculate unclaimed idle CPU. Examples

* criticalstat:     Trace and report long atomic critical sections in the kernel. Examples

* dbslower:     Trace MySQL/PostgreSQL queries slower than a threshold.

* dbstat:     Summarize MySQL/PostgreSQL query latency as a histogram.

* dcsnoop:     Trace directory entry cache (dcache) lookups.

* dcstat:     Directory entry cache (dcache) stats.

* deadlock:     Detect potential deadlocks on a running process.

* drsnoop:     Trace direct reclaim events with PID and latency.

* execsnoop:     Trace new processes via exec() syscalls.

* exitsnoop:     Trace process termination (exit and fatal signals).

* ext4dist:     Summarize ext4 operation latency distribution as a histogram.

* ext4slower:     Trace slow ext4 operations.

* filelife:     Trace the lifespan of short-lived files.

* fileslower:     Trace slow synchronous file reads and writes.

* filetop:     File reads and writes by filename and process. Top for files.

* funccount:     Count kernel function calls.

* funclatency:     Time functions and show their latency distribution.

* funcslower:     Trace slow kernel or user function calls.

* gethostlatency:     Show latency for getaddrinfo/gethostbyname[2] calls.

* hardirqs:     Measure hard IRQ (hard interrupt) event time.

* inject:     Targeted error injection with call chain and predicates

* killsnoop:     Trace signals issued by the kill() syscall.

* klockstat:     Traces kernel mutex lock events and display locks statistics.

* llcstat:     Summarize CPU cache references and misses by process.

* mdflush:     Trace md flush events.

* memleak:     Display outstanding memory allocations to find memory leaks.

* mountsnoop:     Trace mount and umount syscalls system-wide.

* mysqld_qslower:     Trace MySQL server queries slower than a threshold.

* nfsslower:     Trace slow NFS operations.

* nfsdist:     Summarize NFS operation latency distribution as a histogram.

* offcputime:     Summarize off-CPU time by kernel stack trace.

* offwaketime:     Summarize blocked time by kernel off-CPU stack and waker stack.

* oomkill:     Trace the out-of-memory (OOM) killer.

* opensnoop:     Trace open() syscalls.

* pidpersec:     Count new processes (via fork).

* profile:     Profile CPU usage by sampling stack traces at a timed interval.

* reset-trace:     Reset the state of tracing. Maintenance tool only.

* runqlat:     Run queue (scheduler) latency as a histogram.

* runqlen:     Run queue length as a histogram.

* runqslower:     Trace long process scheduling delays.

* shmsnoop:     Trace System V shared memory syscalls.

* sofdsnoop:     Trace FDs passed through unix sockets.

* slabratetop:     Kernel SLAB/SLUB memory cache allocation rate top.

* softirqs:     Measure soft IRQ (soft interrupt) event time.

* solisten:     Trace TCP socket listen.

* sslsniff:     Sniff OpenSSL written and readed data.

* stackcount:     Count kernel function calls and their stack traces.

* syncsnoop:     Trace sync() syscall.

* syscount:     Summarize syscall counts and latencies.

* tcpaccept:     Trace TCP passive connections (accept()).

* tcpconnect:     Trace TCP active connections (connect()).

* tcpconnlat:     Trace TCP active connection latency (connect()).

* tcpdrop:     Trace kernel-based TCP packet drops with details.

* tcplife:     Trace TCP sessions and summarize lifespan.

* tcpretrans:     Trace TCP retransmits and TLPs.

* tcpstates:     Trace TCP session state changes with durations.

* tcpsubnet:     Summarize and aggregate TCP send by subnet.

* tcptop:     Summarize TCP send/recv throughput by host. Top for TCP.

* tcptracer:     Trace TCP established connections (connect(), accept(), close()).

* tplist:     Display kernel tracepoints or USDT probes and their formats.

* trace:     Trace arbitrary functions, with filters.

* ttysnoop:     Watch live output from a tty or pts device.

* ucalls:     Summarize method calls or Linux syscalls in high-level languages.

* uflow:     Print a method flow graph in high-level languages.

* ugc:     Trace garbage collection events in high-level languages.

* uobjnew:     Summarize object allocation events by object type and number of bytes allocated.

* ustat:     Collect events such as GCs, thread creations, object allocations, exceptions and more in high-level languages.

* uthreads:     Trace thread creation events in Java and raw pthreads.

* vfscount vfscount.c:     Count VFS calls.

* vfsstat vfsstat.c:     Count some VFS calls, with column output.

* wakeuptime:     Summarize sleep to wakeup time by waker kernel stack.

* xfsdist:     Summarize XFS operation latency distribution as a histogram.

* xfsslower:     Trace slow XFS operations.

* zfsdist:     Summarize ZFS operation latency distribution as a histogram.

* zfsslower:     Trace slow ZFS operations.


### 推荐

![img.png](image/img3.png)
### 文档
* https://github.com/iovisor/bcc/blob/master/docs/tutorial.md
* https://www.brendangregg.com/ebpf.html
