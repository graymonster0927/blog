# nginx

> [原文](https://www.nginx.com/blog/inside-nginx-how-we-designed-for-performance-scale/)

> [The Architecture of Open Source Applications (Volume 2)nginx](https://aosabook.org/en/v2/nginx.html)

> [Socket Sharding in NGINX](https://www.nginx.com/blog/socket-sharding-nginx-release-1-9-1/)

NGINX在网络性能方面处于领先地位，这完全归功于软件的设计方式。尽管许多web服务器和应用程序服务器使用简单的线程或基于进程的架构，但NGINX凭借复杂的事件驱动架构脱颖而出，使其能够在现代硬件上扩展到数十万个并发连接。

Inside NGINX信息图从高级流程架构深入了解，以说明NGINX如何在单个流程中处理多个连接。这个博客更详细地解释了它是如何工作的。